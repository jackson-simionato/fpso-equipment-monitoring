from src.config import ApplicationConfig
from src.services.statistical_analyzer import StatisticalAnalyzer
from src.services.failure_analyzer import FailureAnalyzer

class AnalyzeStatisticsRoute:
    """
    Route for performing statistical and failure analysis on processed data.

    This class wraps the statistical and failure analysis steps, providing a unified
    interface for generating summary statistics, mean differences, failure event summaries,
    and failure duration analysis.

    Args:
        stat_analyzer (StatisticalAnalyzer): Instance for statistical analysis.
        failure_analyzer (FailureAnalyzer): Instance for failure event analysis.
        config (ApplicationConfig): Application configuration.
    """
    def __init__(self, stat_analyzer: StatisticalAnalyzer, failure_analyzer: FailureAnalyzer, config):
        """
        Initialize the AnalyzeStatisticsRoute.

        Args:
            stat_analyzer (StatisticalAnalyzer): StatisticalAnalyzer instance.
            failure_analyzer (FailureAnalyzer): FailureAnalyzer instance.
            config (ApplicationConfig): Application configuration.
        """
        self.stat_analyzer = stat_analyzer or StatisticalAnalyzer()
        self.failure_analyzer = failure_analyzer or FailureAnalyzer()
        self.config = config or ApplicationConfig()

    def run(self, processed_df):
        """
        Run statistical and failure analysis on the processed DataFrame.

        Args:
            processed_df (pd.DataFrame): The processed data to analyze.

        Returns:
            dict: Dictionary containing summary report, mean difference by target,
                  failure events summary, and failure duration statistics.
        """
        print("\n2. STATISTICAL ANALYSIS...")
        summary_report = self.stat_analyzer.create_stats_summary(processed_df, self.config.sensor_columns)
        print(summary_report)
        mean_diff = self.stat_analyzer.mean_difference_by_target(processed_df, self.config.sensor_columns)
        print("\nTop sensor difference by target:", mean_diff.idxmax())

        print("\n3. FAILURE-SPECIFIC ANALYSIS...")
        failure_summary = self.failure_analyzer.get_failure_events_summary(processed_df)
        print(f"\nFailure Events Summary:\n{failure_summary.head()}")
        failure_duration_stats = self.failure_analyzer.analyze_failure_durations(processed_df)
        print(f"\nFailure Duration Analysis:\n{failure_duration_stats}")

        return {
            "summary_report": summary_report,
            "mean_diff": mean_diff,
            "failure_summary": failure_summary,
            "failure_duration_stats": failure_duration_stats
        }