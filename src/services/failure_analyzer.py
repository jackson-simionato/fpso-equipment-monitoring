import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


class FailureAnalyzer:
    """
    A class specialized in analyzing failure patterns, events, and characteristics
    in FPSO equipment monitoring data.
    """
    
    def __init__(self):
        """Initialize the FailureAnalyzer."""
        pass
    
    # ================================
    # Failure Event Detection and Indexing
    # ================================

    def create_failure_index(self, df: pd.DataFrame, fail_col: str = 'Fail', failure_id_col: str = 'Fail_index') -> pd.DataFrame:
        """
        Create failure event indices for consecutive failure periods.
        
        Args:
            df (pd.DataFrame): DataFrame with failure data
            fail_col (str): Name of the failure boolean column
            failure_id_col (str): Name of the column to store failure event indices
        Returns:
            pd.DataFrame: DataFrame with added 'Fail_index' column
        """
        df_indexed = df.copy()
        
        # Initialize failure index column
        df_indexed[failure_id_col] = 0
        
        # Track failure events
        in_failure = False
        current_fail_index = 0
        
        for idx in df_indexed.index:
            if df_indexed.loc[idx, fail_col]:
                if not in_failure:
                    # Start of new failure event
                    current_fail_index += 1
                    in_failure = True
                df_indexed.loc[idx, failure_id_col] = current_fail_index
            else:
                in_failure = False
                
        return df_indexed
    
    def get_failure_events_summary(self, df: pd.DataFrame,
                                   target_col: str = 'Fail',
                                   failure_index_col: str = 'Fail_index') -> pd.DataFrame:
        """
        Get summary statistics for each failure event.
        
        Args:
            df (pd.DataFrame): DataFrame with 'Fail_index' column
        Returns:
            pd.DataFrame: Summary of failure events (Fail_index, Start_Cycle, End_Cycle, Duration)
        """
        failure_data = df[df[target_col] == True].copy()
        if len(failure_data) == 0:
            return pd.DataFrame()
        # Calculate event statistics: start, end, duration
        event_summary = failure_data.groupby(failure_index_col).agg(
            Start_Cycle=('Cycle', 'min'),
            End_Cycle=('Cycle', 'max'),
            Duration=('Cycle', 'count')
        )
        return event_summary.reset_index()
    
    # ================================
    # Failure Pattern Analysis
    # ================================
    
    def analyze_failure_durations(self, df: pd.DataFrame,
                                  target_col = 'Fail',
                                  failure_index_col = 'Fail_index') -> Dict[str, Any]:
        """
        Analyze the duration patterns of failure events.
        
        Args:
            df (pd.DataFrame): DataFrame with failure event data
            target_col (str): Name of the failure boolean column
            failure_index_col (str): Name of the failure index column
            
        Returns:
            Dict[str, Any]: Duration analysis results
        """
        event_summary = self.get_failure_events_summary(df, target_col, failure_index_col)
        
        if len(event_summary) == 0:
            return {'message': 'No failure events found'}
        
        durations = event_summary['Duration']
        
        analysis = {
            'total_events': len(event_summary),
            'short_failures': len(durations[durations <= 3]),
            'long_failures': len(durations[durations > 3]),
            'event_summary': event_summary
        }
        
        return analysis
    
    def analyze_sensor_behavior_during_failures(self, df: pd.DataFrame, 
                                              sensor_columns: List[str],
                                              target_col: str = 'Fail',
                                              failure_index_col: str = 'Fail_index') -> pd.DataFrame:
        """
        Analyze sensor behavior statistics for each failure event.
        
        Args:
            df (pd.DataFrame): DataFrame with failure events
            sensor_columns (List[str]): List of sensor column names
            
        Returns:
            pd.DataFrame: Sensor statistics by failure event
        """
        failure_data = df[df[target_col] == True].copy()

        if len(failure_data) == 0:
            return pd.DataFrame()
        
        # Calculate statistics for each failure event
        agg_dict = {col: ['mean', 'max', 'std'] for col in sensor_columns}
        agg_dict[failure_index_col] = 'count'  # Count of cycles per event

        sensor_stats = failure_data.groupby(failure_index_col).agg(agg_dict)
        sensor_stats.fillna(0, inplace=True)  # Fill NaN values with 0 for better analysis

        return sensor_stats
    
    def cluster_failure_events(self, df: pd.DataFrame, sensor_columns: List[str], 
                              target_col: str = 'Fail', 
                              failure_index_col: str = 'Fail_index',
                              n_clusters: int = 2, random_state: int = 42) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Cluster failure events based on sensor characteristics.
        
        Args:
            df (pd.DataFrame): DataFrame with failure events
            sensor_columns (List[str]): Sensor columns for clustering
            n_clusters (int): Number of clusters
            random_state (int): Random state for reproducibility
            
        Returns:
            Tuple[pd.DataFrame, np.ndarray]: Sensor stats with clusters, cluster labels
        """
        sensor_stats = self.analyze_sensor_behavior_during_failures(df, sensor_columns, target_col, failure_index_col)
        
        if len(sensor_stats) == 0:
            return pd.DataFrame(), np.array([])
        
        # Prepare data for clustering
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(sensor_stats.fillna(0))
        
        # Perform clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
        cluster_labels = kmeans.fit_predict(scaled_features)
        
        # Add cluster labels to results
        sensor_stats_clustered = sensor_stats.copy()
        sensor_stats_clustered['Cluster'] = cluster_labels
        
        return sensor_stats_clustered, cluster_labels
    
    # ================================
    # Pre-failure Analysis
    # ================================
    
    def analyze_pre_failure_patterns(self, df: pd.DataFrame, sensor_columns: List[str],
                                   buffer_before: int = 5) -> Dict[str, Any]:
        """
        Analyze sensor patterns before failure events.
        
        Args:
            df (pd.DataFrame): DataFrame with failure data
            sensor_columns (List[str]): Sensor columns to analyze
            buffer_before (int): Number of cycles before failure to analyze
            
        Returns:
            Dict[str, Any]: Pre-failure analysis results
        """
        failure_events = self.get_failure_events_summary(df)
        
        if len(failure_events) == 0:
            return {'message': 'No failure events found'}
        
        pre_failure_data = []
        
        for _, event in failure_events.iterrows():
            start_cycle = event['Start_Cycle']
            
            # Get pre-failure buffer
            pre_start = max(start_cycle - buffer_before, df.index.min())
            pre_end = start_cycle - 1
            
            if pre_start <= pre_end:
                pre_data = df.loc[pre_start:pre_end, sensor_columns + ['Preset_1', 'Preset_2']]
                if len(pre_data) > 0:
                    pre_data = pre_data.copy()
                    pre_data['Event_ID'] = event['Fail_index']
                    pre_data['Cycles_Before_Failure'] = start_cycle - pre_data.index
                    pre_failure_data.append(pre_data)
        
        if not pre_failure_data:
            return {'message': 'No pre-failure data available'}
        
        # Combine all pre-failure data
        combined_pre_failure = pd.concat(pre_failure_data, ignore_index=False)
        
        # Analyze trends
        analysis = {
            'pre_failure_data': combined_pre_failure,
            'sensor_trends': {},
            'preset_patterns': {}
        }
        
        # Calculate sensor trends for each event
        for sensor in sensor_columns:
            event_trends = []
            for event_id in combined_pre_failure['Event_ID'].unique():
                event_data = combined_pre_failure[combined_pre_failure['Event_ID'] == event_id]
                if len(event_data) >= 2:
                    # Calculate trend (simple slope)
                    x = event_data['Cycles_Before_Failure'].values
                    y = event_data[sensor].values
                    if len(x) > 1:
                        trend = np.polyfit(x, y, 1)[0]  # Slope
                        event_trends.append({
                            'Event_ID': event_id,
                            'Trend': trend,
                            'Mean_Value': y.mean(),
                            'Final_Value': y[-1] if len(y) > 0 else np.nan
                        })
            
            analysis['sensor_trends'][sensor] = pd.DataFrame(event_trends)
        
        # Analyze preset patterns before failures
        preset_patterns = combined_pre_failure.groupby('Event_ID')[['Preset_1', 'Preset_2']].agg({
            'Preset_1': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else x.iloc[-1],
            'Preset_2': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else x.iloc[-1]
        })
        analysis['preset_patterns'] = preset_patterns
        
        return analysis
    
    def generate_failure_report(self, df: pd.DataFrame, sensor_columns: List[str]) -> str:
        """
        Generate a comprehensive failure analysis report.
        
        Args:
            df (pd.DataFrame): DataFrame with failure data and 'Fail_index' column
            sensor_columns (List[str]): List of sensor column names
            
        Returns:
            str: Formatted failure report
        """
        # Get basic failure statistics
        failure_analysis = self.analyze_failure_patterns(df, sensor_columns)
        duration_analysis = self.analyze_failure_durations(df)
        
        report = []
        report.append("=" * 50)
        report.append("FAILURE ANALYSIS REPORT")
        report.append("=" * 50)
        
        # Basic statistics
        report.append(f"\nBASIC FAILURE STATISTICS:")
        report.append(f"- Total failures: {failure_analysis['failure_count']}")
        report.append(f"- Failure rate: {failure_analysis['failure_rate']:.2%}")
        report.append(f"- Total cycles analyzed: {failure_analysis['total_cycles']}")
        
        # Duration analysis
        if 'duration_stats' in duration_analysis:
            stats = duration_analysis['duration_stats']
            report.append(f"\nFAILURE DURATION ANALYSIS:")
            report.append(f"- Total failure events: {duration_analysis['total_events']}")
            report.append(f"- Average duration: {stats['mean']:.1f} cycles")
            report.append(f"- Median duration: {stats['median']:.1f} cycles")
            report.append(f"- Shortest failure: {stats['min']:.0f} cycles")
            report.append(f"- Longest failure: {stats['max']:.0f} cycles")
            report.append(f"- Short failures (≤3 cycles): {duration_analysis['short_failures']}")
            report.append(f"- Long failures (>3 cycles): {duration_analysis['long_failures']}")
        
        # Top sensor indicators
        if len(failure_analysis['sensor_differences']) > 0:
            top_sensors = failure_analysis['sensor_differences'].head(3)
            report.append(f"\nTOP FAILURE INDICATORS:")
            for sensor, diff in top_sensors.items():
                report.append(f"- {sensor}: {diff:.2f} difference during failures")
        
        report.append("\n" + "=" * 50)
        
        return "\n".join(report)
    
    # ================================
    # Domain-Specific Analysis Methods
    # ================================
    
    def analyze_failure_patterns(self, df: pd.DataFrame, sensor_columns: List[str]) -> dict:
        """
        Analyze patterns in failure data including distributions and trends.
        
        Args:
            df (pd.DataFrame): DataFrame containing failure data
            sensor_columns (List[str]): List of sensor column names
            
        Returns:
            dict: Dictionary containing various failure analysis results
        """
        results = {}
        
        # Import statistical analyzer for helper methods
        from .statistical_analyzer import StatisticalAnalyzer
        stats_analyzer = StatisticalAnalyzer()
        
        # Basic failure statistics
        results['failure_count'] = df['Fail'].sum()
        results['failure_rate'] = df['Fail'].mean()
        results['total_cycles'] = len(df)
        
        # Sensor behavior during failures
        results['sensor_differences'] = stats_analyzer.mean_difference_by_target(
            df, sensor_columns, 'Fail'
        ).sort_values(ascending=False)
        
        # Failure statistics by sensor
        failure_stats = stats_analyzer.create_stats_summary(
            df, target_col='Fail', agg_funcs=['mean', 'median', 'std', 'max', 'min']
        ).drop(columns=['Fail'], errors='ignore')
        results['failure_sensor_stats'] = failure_stats
        
        return results
    
    def analyze_preset_risk(self, df: pd.DataFrame) -> dict:
        """
        Analyze risk levels associated with different preset combinations.
        
        Args:
            df (pd.DataFrame): DataFrame containing preset and failure data
            
        Returns:
            dict: Dictionary containing preset risk analysis
        """
        results = {}
        
        # Failure rates by individual presets
        results['preset_1_failure_rates'] = df.groupby('Preset_1')['Fail'].mean().sort_values(ascending=False)
        results['preset_2_failure_rates'] = df.groupby('Preset_2')['Fail'].mean().sort_values(ascending=False)
        
        # Failure rates by preset combinations
        combo_failure_rates = df.groupby(['Preset_1', 'Preset_2'])['Fail'].agg(['mean', 'count']).reset_index()
        combo_failure_rates = combo_failure_rates[combo_failure_rates['count'] >= 5]  # Filter for meaningful sample sizes
        combo_failure_rates = combo_failure_rates.sort_values('mean', ascending=False)
        results['combo_failure_rates'] = combo_failure_rates
        
        # High-risk combinations (above average failure rate)
        avg_failure_rate = df['Fail'].mean()
        high_risk_combos = combo_failure_rates[combo_failure_rates['mean'] > avg_failure_rate * 1.5]
        results['high_risk_combinations'] = high_risk_combos
        
        return results