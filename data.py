import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Union, Any

class PTL:
    """
    Put to Light (PTL) System Optimization Class
    
    Handles data preprocessing, order time computation, and zone time analysis
    for Put to Light warehouse operations.
    """
    def __init__(self, option: int = 1):
        """
        Initialize PTL system with data configuration.
        
        :param option: Selection of data configuration (1-6)
        """
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)

        self._option_data = {
            1: 'Data_40_Salidas_composicion_zonas_heterogeneas.xlsx',
            2: 'Data_40_Salidas_composicion_zonas_homogeneas.xlsx',
            3: 'Data_60_Salidas_composicion_zonas_heterogeneas.xlsx',
            4: 'Data_60_Salidas_composicion_zonas_homogeneas.xlsx',
            5: 'Data_80_Salidas_composicion_zonas_heterogeneas.xlsx',
            6: 'Data_80_Salidas_composicion_zonas_homogeneas.xlsx'
        }
        
        self._validate_option(option)
        self._option = option
        
        # Initialize data attributes
        self._initialize_data_attributes()

    def _validate_option(self, option: int):
        """
        Validate the selected data configuration option.
        
        :param option: Option number to validate
        :raises ValueError: If option is not in valid range
        """
        if option not in self._option_data:
            raise ValueError(f"Invalid option: {option}. Must be between 1 and 6.")

    def _initialize_data_attributes(self):
        """
        Initialize data attributes with default values.
        """
        self._data_sheets = None
        self._orders_set = []
        self._zones_set = []
        self._departures_set = []
        self._skus_set = []
        self._workers_set = []
        
        self._v = 0.0  # Speed of worker
        self._zn = 0.0  # Number of zones
        
        self._n_departures_per_zone = {}
        self._n_orders = 0
        self._n_departures = 0
        self._departures_per_zone = {}
        self._departure_time = None
        self._total_departure_time = {}
        self._skus_per_order = {}
        self._n_skus_per_order = {}
        self._sku_time = None
        self._solution = {}
        self._solution_correspondence = {}
        self._order_times = {}
        self._zone_times = []
        
        # Optimization: Pre-computed values
        self._sku_time_sums = {}
        self._departure_time_lookup = {}
        self._zone_lookup = {}

    def load_data(self):
        """
        Load and preprocess data from Excel sheets.
        """
        try:
            self._data_sheets = pd.ExcelFile(self._option_data[self._option])
            self._load_sets()
            self._load_parameters()
            self._load_departure_parameters()
            self._load_sku_parameters()
            self._initialize_solution()
            # Optimization: Pre-compute frequently used values
            self._precompute_values()
        except FileNotFoundError:
            self.logger.error(f"File not found: {self._option_data[self._option]}")
            raise
        except Exception as e:
            self.logger.error(f"Error loading data: {e}")
            raise

    def _load_sets(self):
        """
        Load set data from Excel sheets.
        """
        sets_to_load = [
            ('orders', 'Pedidos'),
            ('zones', 'Zonas'),
            ('departures', 'Salidas'),
            ('skus', 'SKU'),
            ('workers', 'Trabajadores')
        ]
        
        for attr, sheet in sets_to_load:
            df = pd.read_excel(self._data_sheets, sheet, index_col=0)
            setattr(self, f'_{attr}_set', list(df.index))

    def _load_parameters(self):
        """
        Load speed and zone parameters.
        """
        parameters = pd.read_excel(self._data_sheets, 'Parametros', index_col=0).values.tolist()
        self._v = parameters[0][0]  # Worker speed
        self._zn = parameters[0][1]  # Number of zones

    def _load_departure_parameters(self):
        """
        Load departure-related parameters.
        """
        # Number of departures in each zone
        self._n_departures_per_zone = pd.read_excel(
            self._data_sheets, 
            'Salidas_en_cada_zona', 
            index_col=0
        ).iloc[:, 0].to_dict()

        self._n_orders = len(self._orders_set)
        self._n_departures = len(self._departures_set)

        # Departures per zone (binary mapping)
        departures_per_zone = pd.read_excel(
            self._data_sheets, 
            'Salidas_pertenece_zona', 
            index_col=0
        )

        self._departures_per_zone = {
            col: departures_per_zone.index[departures_per_zone[col] == 1][0]
            for col in departures_per_zone.columns
        }

        # Departure times
        departure_distance = pd.read_excel(
            self._data_sheets, 
            'Tiempo_salida', 
            index_col=0
        )

        self._departure_time = 2 * departure_distance / self._v

        # Total processing time for each departure
        self._total_departure_time = np.sum(self._departure_time, axis=0).to_dict()

    def _load_sku_parameters(self):
        """
        Load SKU-related parameters.
        """
        # SKUs per order
        self._skus_per_order = pd.read_excel(
            self._data_sheets, 
            'SKU_pertenece_pedido', 
            index_col=0
        ).apply(
            lambda x: x[x == 1].index.tolist(), 
            axis=1
        ).to_dict()

        # Number of SKUs per order
        self._n_skus_per_order = {
            order: len(skus) 
            for order, skus in self._skus_per_order.items()
        }

        # SKU processing times
        self._sku_time = pd.read_excel(
            self._data_sheets, 
            'Tiempo_SKU', 
            index_col=0
        )

    def _precompute_values(self):
        """
        Optimization: Pre-compute frequently used values to avoid repeated calculations.
        """
        # Pre-compute sum of SKU processing times for each order
        self._sku_time_sums = {}
        for order in self._orders_set:
            self._sku_time_sums[order] = self._sku_time.loc[order].sum()
        
        # Create a faster lookup for departure times
        self._departure_time_lookup = {}
        for departure in self._departures_set:
            zone = self._departures_per_zone[departure]
            if zone in self._departure_time.index and departure in self._departure_time.columns:
                self._departure_time_lookup[departure] = self._departure_time.loc[zone, departure]
        
        # Create a fast lookup for zone by departure
        self._zone_lookup = self._departures_per_zone.copy()
        
        # Pre-compute zone index mapping for faster zone time updates
        self._zone_indices = {zone: idx for idx, zone in enumerate(self._zones_set)}

    def _initialize_solution(self):
        """
        Create an initial solution with orders assigned to departures.
        """
        self._solution = dict(zip(
            self._orders_set, 
            self._departures_set[:len(self._orders_set)]
        ))
    
    def check_solution(self) -> bool:
        """
        Check if the current solution is valid.
        
        :return: True if the solution is valid, False otherwise
        """
        return len(self._solution) == len(self._orders_set) and all(order in self._solution for order in self._orders_set) and len(set(self._solution.values())) == len(set(self._solution.keys()))
    
    def compute_order_time(self, order: str, departure: str) -> float:
        """
        Compute total time for processing a specific order.
        
        :param order: Order identifier
        :param departure: Departure identifier
        :return: Total time for order processing
        """
        # Optimization: Use pre-computed values instead of dataframe lookups
        sku_processing_time = self._sku_time_sums[order]
        departure_processing_time = self._departure_time_lookup[departure]
        total_processing_time = departure_processing_time * self._n_skus_per_order[order]
        
        return sku_processing_time + total_processing_time

    def compute_order_times(self) -> Dict[str, float]:
        """
        Compute processing times for all orders.
        
        :return: Dictionary of order times
        """
        # Optimization: Use pre-allocated dictionary and avoid method call overhead
        order_times = {}
        for order in self._orders_set:
            departure = self._solution[order]
            order_times[order] = self._sku_time_sums[order] + (self._departure_time_lookup[departure] * self._n_skus_per_order[order])
        
        self._order_times = order_times
        return order_times

    def compute_total_zone_time(self) -> List[float]:
        """
        Compute total processing time for each zone.
        
        :return: List of zone processing times
        """
        # Optimization: Use pre-allocated dictionary with zone indices
        zone_times = {zone: 0.0 for zone in self._zones_set}
        
        # If order times haven't been computed, compute them
        if not self._order_times:
            self.compute_order_times()
        
        # Sum up times by zone
        for order, time in self._order_times.items():
            departure = self._solution[order]
            zone = self._zone_lookup[departure]
            zone_times[zone] += time
        
        # Convert to list in the correct order
        self._zone_times = [zone_times[zone] for zone in self._zones_set]
        
        return self._zone_times

    def update_solution(self, order: str, new_departure: str):
        """
        Optimization: Update a single order assignment and efficiently update related metrics.
        
        :param order: Order identifier
        :param new_departure: New departure identifier
        """
        old_departure = self._solution[order]
        
        # Skip if no change
        if old_departure == new_departure:
            return
        
        # Update solution
        self._solution[order] = new_departure
        
        # Update order time if already computed
        if self._order_times:
            old_time = self._order_times[order]
            new_time = self.compute_order_time(order, new_departure)
            self._order_times[order] = new_time
            
            # Update zone times if already computed
            if self._zone_times:
                old_zone = self._zone_lookup[old_departure]
                new_zone = self._zone_lookup[new_departure]
                
                # Get zone indices
                old_idx = self._zone_indices[old_zone]
                new_idx = self._zone_indices[new_zone]
                
                # Update zone times
                self._zone_times[old_idx] -= old_time
                self._zone_times[new_idx] += new_time

    def save_solution(self, filename: str):
        """
        Save the current solution to an Excel file.

        :param filename: Name of the Excel file to save
        """
        # Ensure zone times are computed
        if not self._zone_times:
            self.compute_total_zone_time()
            
        # Obtener el índice de la zona con el máximo tiempo
        max_index = self._zone_times.index(max(self._zone_times))

        # Crear el primer DataFrame
        sheet1 = pd.DataFrame({
            'Instancia': [self._option_data[self._option]],
            'Zona': [self._zones_set[max_index]],
            'Maximo': [max(self._zone_times)]
        })

        # Crear el segundo DataFrame
        sheet2 = pd.DataFrame({
            'Pedido': list(self._solution.keys()),
            'Salida': [i for i in self._solution.values()]
        })

        # Crear el tercer DataFrame
        sheet3 = pd.DataFrame({
            'Zona': list(self._zones_set),
            'Tiempo': list(self._zone_times)
        })

        # Guardar los DataFrames en un archivo Excel
        with pd.ExcelWriter(filename) as writer:
            sheet1.to_excel(writer, sheet_name='Resumen', index=False)
            sheet2.to_excel(writer, sheet_name='Solucion', index=False)
            sheet3.to_excel(writer, sheet_name='Metricas', index=False)
    
    # Getter Methods
    def get_option(self) -> int:
        """
        Get the current data configuration option.
        
        :return: Selected option number
        """
        return self._option

    def get_set(self, set_name: str) -> List[str]:
        """
        Retrieve a specific set of data.
        
        :param set_name: Name of the set (orders, zones, departures, skus, workers)
        :return: List of items in the specified set
        :raises ValueError: If the set name is invalid
        """
        set_mapping = {
            'orders': self._orders_set,
            'zones': self._zones_set,
            'departures': self._departures_set,
            'skus': self._skus_set,
            'workers': self._workers_set
        }
        
        if set_name.lower() not in set_mapping:
            raise ValueError(f"Invalid set name: {set_name}")
        
        return set_mapping[set_name.lower()]

    def get_parameter(self, param_name: str) -> Union[float, Dict]:
        """
        Retrieve a specific parameter or parameters.
        
        :param param_name: Name of the parameter to retrieve
        :return: Parameter value or dictionary of parameters
        :raises ValueError: If the parameter name is invalid
        """
        param_mapping = {
            'file_name': self._option_data[self._option],
            'worker_speed': self._v,
            'n_zones': self._zn,
            'n_departures_per_zone': self._n_departures_per_zone,
            'n_orders': self._n_orders,
            'n_departures': self._n_departures,
            'departures_per_zone': self._departures_per_zone,
            'solution': self._solution,
            'order_times': self._order_times,
            'zone_times': self._zone_times,
            'skus_per_order': self._skus_per_order,
            'n_skus_per_order': self._n_skus_per_order,
            'sku_time': self._sku_time,
            'departure_time': self._departure_time,
            'total_departure_time': self._total_departure_time
        }
        
        if param_name.lower() not in param_mapping:
            raise ValueError(f"Invalid parameter name: {param_name}")
        
        return param_mapping[param_name.lower()]

    # Setter Methods
    def set_option(self, option: int):
        """
        Set a new data configuration option.
        
        :param option: New option number to set
        """
        self._validate_option(option)
        self._option = option
        self.load_data()

    def set_set(self, set_name: str, new_set: List[str]):
        """
        Update a specific set of data.
        
        :param set_name: Name of the set to update
        :param new_set: New list of items for the set
        :raises ValueError: If the set name is invalid
        """
        set_mapping = {
            'orders': '_orders_set',
            'zones': '_zones_set',
            'departures': '_departures_set',
            'skus': '_skus_set',
            'workers': '_workers_set'
        }
        
        if set_name.lower() not in set_mapping:
            raise ValueError(f"Invalid set name: {set_name}")
        
        setattr(self, set_mapping[set_name.lower()], new_set)

    def set_parameter(self, param_name: str, value: Any):
        """
        Update a specific parameter.
        
        :param param_name: Name of the parameter to update
        :param value: New value for the parameter
        :raises ValueError: If the parameter name is invalid
        """
        param_mapping = {
            'worker_speed': '_v',
            'zones': '_zn',
            'n_departures_per_zone': '_n_departures_per_zone',
            'departures_per_zone': '_departures_per_zone',
            'solution': '_solution',
            'order_times': '_order_times',
            'zone_times': '_zone_times'
        }
        
        if param_name.lower() not in param_mapping:
            raise ValueError(f"Invalid parameter name: {param_name}")
        
        setattr(self, param_mapping[param_name.lower()], value)

    def __repr__(self):
        """
        Provide a string representation of the PTL instance.
        
        :return: String with key information about the PTL instance
        """
        return (f"PTL System (Option {self._option})\n"
                f"Orders: {len(self._orders_set)}\n"
                f"Zones: {len(self._zones_set)}\n"
                f"Departures: {len(self._departures_set)}\n"
                f"SKUs: {len(self._skus_set)}\n"
                f"Workers: {len(self._workers_set)}")
    
    def copy(self):
        """
        Create a shallow copy of the PTL instance.
        
        :return: A new PTL instance with the same data
        """
        new_instance = PTL(self._option)
        new_instance.logger = self.logger
        new_instance._v = self._v
        new_instance._zn = self._zn
        new_instance._n_departures_per_zone = self._n_departures_per_zone.copy()
        new_instance._n_orders = self._n_orders
        new_instance._n_departures = self._n_departures
        new_instance._departures_per_zone = self._departures_per_zone.copy()
        new_instance._departure_time = self._departure_time.copy()
        new_instance._total_departure_time = self._total_departure_time.copy()
        new_instance._skus_per_order = self._skus_per_order.copy()
        new_instance._n_skus_per_order = self._n_skus_per_order.copy()
        new_instance._sku_time = self._sku_time.copy()
        new_instance._solution = self._solution.copy()
        new_instance._solution_correspondence = self._solution_correspondence.copy()
        new_instance._order_times = self._order_times.copy()
        new_instance._zone_times = self._zone_times.copy()
        new_instance._option_data = self._option_data.copy()
        new_instance._option = self._option
        new_instance._orders_set = self._orders_set.copy()
        new_instance._zones_set = self._zones_set.copy()
        new_instance._departures_set = self._departures_set.copy()
        new_instance._skus_set = self._skus_set.copy()
        new_instance._workers_set = self._workers_set.copy()
        
        # Copy optimization data structures
        new_instance._sku_time_sums = self._sku_time_sums.copy()
        new_instance._departure_time_lookup = self._departure_time_lookup.copy()
        new_instance._zone_lookup = self._zone_lookup.copy()
        new_instance._zone_indices = self._zone_indices.copy()
    
        return new_instance