"""
Visualization module for data analysis exploration.

This module provides plotting functions using seaborn for exploratory data analysis
of FPSO equipment monitoring data.
"""

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import seaborn as sns
from typing import Optional, List, Tuple, Union


class DataPlotter:
    """
    A class for creating visualizations for exploratory data analysis.
    """
    def __init__(self,
                figsize=(10, 6)):
        """
        Initialize the DataPlotter with a dark style and pink highlight.
        """
        self.figsize = figsize

        # Basic Seaborn dark style
        sns.set_style("darkgrid")
        sns.set_palette(["#E61E5C"])  # brand pink

        # Minimal Matplotlib adjustments
        plt.rcParams["figure.facecolor"] = "#1e1e1e"
        plt.rcParams["axes.facecolor"] = "#1e1e1e"
        plt.rcParams["axes.edgecolor"] = "#FFFFFF"
        plt.rcParams["axes.labelcolor"] = "#FFFFFF"
        plt.rcParams["xtick.color"] = "#FFFFFF"
        plt.rcParams["ytick.color"] = "#FFFFFF"
        plt.rcParams["text.color"] = "#FFFFFF"
    
    def plot_histogram(self, 
                      data: pd.DataFrame, 
                      column: str, 
                      bins: Union[int, str] = 'auto',
                      kde: bool = False,
                      title: Optional[str] = None,
                      xlabel: Optional[str] = None,
                      ylabel: str = 'Frequency',
                      figsize: Optional[Tuple[int, int]] = None,
                      alpha: float = 1,
                      bar_width: float = 0.6) -> plt.Figure:
        """
        Create a histogram for a specific column.
        
        Args:
            data: DataFrame containing the data
            column: Column name to plot
            bins: Number of bins or method for determining bins
            kde: Whether to include kernel density estimation
            title: Plot title (defaults to column name)
            xlabel: X-axis label (defaults to column name)
            ylabel: Y-axis label
            figsize: Figure size (width, height)
            alpha: Transparency level
            bar_width: Width of bars for categorical data (0.0 to 1.0)
            
        Returns:
            matplotlib Figure object
        """
        if figsize is None:
            figsize = self.figsize
            
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create histogram
        if data[column].dtype in ['object', 'int', 'bool']:
            sns.countplot(data=data, x=column, ax=ax, alpha=alpha, width=bar_width)
        else:
            sns.histplot(data=data, x=column, bins=bins, kde=kde, alpha=alpha, ax=ax)
        
        # Set labels and title
        if title is None:
            title = f'Distribution of {column}'
        if xlabel is None:
            xlabel = column
            
        ax.set_title(title, fontsize=14)
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        
        # Add grid for better readability
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_multiple_histograms(self, 
                                data: pd.DataFrame, 
                                columns: List[str],
                                bins: Union[int, str] = 'auto',
                                kde: bool = False,
                                cols: int = 2,
                                figsize: Optional[Tuple[int, int]] = None,
                                alpha: float = 1,
                                bar_width: float = 0.6) -> plt.Figure:
        """
        Create multiple histograms in a subplot layout.
        
        Args:
            data: DataFrame containing the data
            columns: List of column names to plot
            bins: Number of bins or method for determining bins
            kde: Whether to include kernel density estimation
            cols: Number of columns in subplot grid
            figsize: Figure size (width, height)
            alpha: Transparency level
            bar_width: Width of bars for categorical data (0.0 to 1.0)
            
        Returns:
            matplotlib Figure object
        """
        n_plots = len(columns)
        rows = (n_plots + cols - 1) // cols  # Calculate number of rows needed
        
        if figsize is None:
            figsize = (6 * cols, 4 * rows)
            
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        
        # Handle case where there's only one subplot
        if n_plots == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes.flatten()
        else:
            axes = axes.flatten()
        
        for i, column in enumerate(columns):
            # Create histogram
            if data[column].dtype in ['object', 'int', 'bool']:
                sns.countplot(data=data, x=column, ax=axes[i], alpha=alpha, width=bar_width)
            else:
                sns.histplot(data=data, x=column, bins=bins, kde=kde, alpha=alpha, ax=axes[i])

            axes[i].set_title(f'Distribution of {column}', fontsize=12)
            axes[i].set_xlabel(column, fontsize=10)
            axes[i].set_ylabel('Frequency', fontsize=10)
            axes[i].grid(True, alpha=0.3)
        
        # Hide empty subplots
        for i in range(n_plots, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        return fig
    
    def plot_histogram_by_category(self, 
                                  data: pd.DataFrame, 
                                  column: str,
                                  category: str,
                                  bins: Union[int, str] = 'auto',
                                  kde: bool = False,
                                  title: Optional[str] = None,
                                  xlabel: Optional[str] = None,
                                  figsize: Optional[Tuple[int, int]] = None,
                                  alpha: float = 1) -> plt.Figure:
        """
        Create histograms split by a categorical variable.
        
        Args:
            data: DataFrame containing the data
            column: Column name to plot (numeric)
            category: Categorical column to split by
            bins: Number of bins or method for determining bins
            kde: Whether to include kernel density estimation
            title: Plot title
            xlabel: X-axis label
            figsize: Figure size (width, height)
            alpha: Transparency level
            
        Returns:
            matplotlib Figure object
        """
        if figsize is None:
            figsize = self.figsize
            
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create histogram with hue
        sns.histplot(data=data, x=column, hue=category, bins=bins, kde=kde, alpha=alpha, ax=ax)
        
        # Set labels and title
        if title is None:
            title = f'Distribution of {column} by {category}'
        if xlabel is None:
            xlabel = column
            
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        
        # Add grid for better readability
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_boxplot(self, 
                    data: pd.DataFrame, 
                    column: str,
                    category: Optional[str] = None,
                    title: Optional[str] = None,
                    figsize: Optional[Tuple[int, int]] = None) -> plt.Figure:
        """
        Create a boxplot for a specific column, optionally grouped by category.
        
        Args:
            data: DataFrame containing the data
            column: Column name to plot (numeric)
            category: Optional categorical column to group by
            title: Plot title
            figsize: Figure size (width, height)
            
        Returns:
            matplotlib Figure object
        """
        if figsize is None:
            figsize = self.figsize
            
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create boxplot
        if category is None:
            sns.boxplot(data=data, y=column, ax=ax, color="#E61E5C", 
                       flierprops=dict(markerfacecolor='white', markeredgecolor='white', markersize=4))
        else:
            sns.boxplot(data=data, x=category, y=column, ax=ax,
                       flierprops=dict(markerfacecolor='white', markeredgecolor='white', markersize=4))
        
        # Set labels and title
        if title is None:
            if category is None:
                title = f'Boxplot of {column}'
            else:
                title = f'{column} by {category}'
                
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_multiple_boxplots(self, 
                              data: pd.DataFrame, 
                              columns: List[str],
                              category: Optional[str] = None,
                              cols: int = 2,
                              figsize: Optional[Tuple[int, int]] = None) -> plt.Figure:
        """
        Create multiple boxplots in a subplot layout.
        
        Args:
            data: DataFrame containing the data
            columns: List of column names to plot (numeric)
            category: Optional categorical column to group by
            cols: Number of columns in subplot grid
            figsize: Figure size (width, height)
            
        Returns:
            matplotlib Figure object
        """
        n_plots = len(columns)
        rows = (n_plots + cols - 1) // cols
        
        if figsize is None:
            figsize = (6 * cols, 4 * rows)
            
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        
        # Handle different subplot cases
        if n_plots == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes.flatten()
        else:
            axes = axes.flatten()
        
        for i, column in enumerate(columns):
            # Create boxplot
            if category is None:
                sns.boxplot(data=data, y=column, ax=axes[i], color="#E61E5C",
                           flierprops=dict(markerfacecolor='white', markeredgecolor='white', markersize=4))
            else:
                sns.boxplot(data=data, x=category, y=column, ax=axes[i],
                           flierprops=dict(markerfacecolor='white', markeredgecolor='white', markersize=4))
            
            # Set title and grid
            axes[i].set_title(f'{column}', fontsize=12, fontweight='bold')
            axes[i].grid(True, alpha=0.3)
        
        # Hide empty subplots
        for i in range(n_plots, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        return fig
    
    def plot_time_series(self, 
                        data: pd.DataFrame, 
                        columns: Union[str, List[str]],
                        x_col: Optional[str] = None,
                        title: Optional[str] = None,
                        figsize: Optional[Tuple[int, int]] = None) -> plt.Figure:
        """
        Create a time series line plot for one or multiple columns.

        Args:
            data: DataFrame containing the data
            columns: Column name(s) to plot (string for single, list for multiple)
            x_col: Column to use for x-axis (if None, uses DataFrame index)
            title: Plot title
            figsize: Figure size (width, height)
        
        Returns:
            matplotlib Figure object
        """
        if figsize is None:
            figsize = self.figsize
            
        fig, ax = plt.subplots(figsize=figsize)
        
        # Convert single column to list for consistent handling
        if isinstance(columns, str):
            columns = [columns]
        
        # Set x-axis data
        x_data = data[x_col] if x_col else data.index
        
        # Plot lines for each column
        for column in columns:
            ax.plot(x_data, data[column], label=column, linewidth=2)
        
        # Set labels and title
        if title is None:
            if len(columns) == 1:
                title = f'Time Series: {columns[0]}'
            else:
                title = 'Time Series Plot'
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel(x_col if x_col else data.index.name, fontsize=12)
        ax.set_ylabel('-'.join(columns), fontsize=12)
        ax.set_ylim(0, data[columns].max().max() * 1.1)  # Set y-limits to 110% of max value
        
        # Add legend if multiple columns
        if len(columns) > 1:
            ax.legend()
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig

    def plot_correlation_heatmap(self, 
                                data: pd.DataFrame, 
                                columns: Optional[List[str]] = None,
                                title: Optional[str] = None,
                                figsize: Optional[Tuple[int, int]] = None) -> plt.Figure:
        """
        Create a correlation heatmap for numeric columns.
    
        Args:
            data: DataFrame containing the data
            columns: List of columns to include (if None, uses all numeric columns)
            title: Plot title
            figsize: Figure size (width, height)
        
        Returns:
            matplotlib Figure object
        """
        # Select columns
        if columns is None:
            numeric_data = data.select_dtypes(float)
        else:
            numeric_data = data[columns]
    
        # Calculate correlation matrix
        corr_matrix = numeric_data.corr()
    
        if figsize is None:
            # Auto-size based on number of variables
            size = max(8, len(corr_matrix.columns) * 0.8)
            figsize = (size, size)
        
        fig, ax = plt.subplots(figsize=figsize)
    
        # Create heatmap
        sns.heatmap(corr_matrix, 
                    annot=True, 
                    fmt='.2f', 
                    cmap='RdBu_r',
                    center=0,
                    square=True,
                    linewidths=0.5,
                    cbar_kws={'shrink': 0.8},
                    ax=ax)
    
        # Set title
        if title is None:
            title = 'Correlation Matrix'
        
        ax.set_title(title, fontsize=14)
    
        plt.tight_layout()
        return fig

    def plot_failure_rate_heatmap(self, 
                                  data: pd.DataFrame, 
                                  row_var: str,
                                  col_var: str,
                                  target_var: str = "Fail",
                                  title: Optional[str] = None,
                                  figsize: Optional[Tuple[int, int]] = None) -> plt.Figure:
        """
        Create a heatmap showing failure rates across two categorical variables.
    
        Args:
            data: DataFrame containing the data
            row_var: Variable for rows (e.g., 'Preset_1')
            col_var: Variable for columns (e.g., 'Preset_2') 
            title: Plot title
            figsize: Figure size (width, height)
        
        Returns:
            matplotlib Figure object
        """
        # Calculate failure rate by grouping variables
        failure_rate = data.groupby([row_var, col_var])[target_var].mean().unstack(fill_value=0)
    
        if figsize is None:
            figsize = self.figsize
        
        fig, ax = plt.subplots(figsize=figsize)
    
        # Create heatmap
        sns.heatmap(failure_rate, 
                    annot=True, 
                    fmt='.2f', 
                    cmap='Reds',
                    square=True,
                    linewidths=0.5,
                    cbar_kws={'shrink': 0.8},
                    ax=ax)
    
        # Set title and labels
        if title is None:
            title = f'Failure Rate by {row_var} and {col_var}'
        
        ax.set_title(title, fontsize=14, fontweight='bold')
    
        plt.tight_layout()
        return fig

    def plot_bar(self, 
                 data: pd.DataFrame, 
                 x: str, 
                 y: str, 
                 top_n: Optional[int] = None,
                 figsize: Optional[Tuple[int, int]] = None,
                 title: Optional[str] = None,
                 xlabel: Optional[str] = None,
                 ylabel: Optional[str] = None,
                 color: Optional[str] = "#E61E5C") -> plt.Figure:
        """
        Create a barplot for any two columns.

        Args:
            data: DataFrame containing the data
            x: Column name for x-axis
            y: Column name for y-axis
            top_n: If set, plot only the top N rows (sorted by y descending)
            figsize: Figure size (width, height)
            title: Plot title
            xlabel: X-axis label
            ylabel: Y-axis label
            color: Bar color

        Returns:
            matplotlib Figure object
        """
        if figsize is None:
            figsize = self.figsize

        plot_data = data.copy()
        if top_n is not None:
            plot_data = plot_data.sort_values(by=y, ascending=False).head(top_n)

        fig, ax = plt.subplots(figsize=figsize)
        sns.barplot(x=x, y=y, data=plot_data, ax=ax, color=color)

        if title is None:
            title = f"Barplot of {y} by {x}"
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel(xlabel or x, fontsize=12)
        ax.set_ylabel(ylabel or y, fontsize=12)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig
    
    ## Functions related to failure
    def plot_failure_timeline(self, df: pd.DataFrame, figsize: Tuple[int, int] = (12, 6), target_col: str = 'Fail') -> plt.Figure:
        """
        Create a timeline plot showing failure events over time.
        
        Args:
            df (pd.DataFrame): DataFrame with failure data
            figsize (Tuple[int, int]): Figure size
            
        Returns:
            plt.Figure: Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create failure state as integer
        failure_over_time = df[target_col].astype(int)
        
        # Plot as bar chart
        colors = ['lightblue' if x == 0 else 'red' for x in failure_over_time]
        ax.bar(failure_over_time.index, failure_over_time, 
               color=colors, alpha=0.7, width=1)
        
        ax.set_xlabel('Operation Cycle')
        ax.set_ylabel('Failure State')
        ax.set_title('Failure Events Over Time')
        ax.set_yticks([0, 1])
        ax.set_yticklabels(['Normal', 'Failure'])
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_failure_summary_heatmap(self, df: pd.DataFrame, sensor_columns: List[str],
                                    target_col: str = 'Fail',
                                    failure_id_col: str = 'Fail_index',
                                    agg_funcs: List[str] = ['mean', 'max', 'std'],
                                    figsize: Tuple[int, int] = (16, 10)) -> plt.Figure:
        """
        Create a heatmap showing failure event sensor statistics.
        
        Args:
            df (pd.DataFrame): DataFrame with failure events
            sensor_columns (List[str]): Sensor columns to analyze
            agg_funcs (List[str]): Aggregation functions
            figsize (Tuple[int, int]): Figure size
            
        Returns:
            plt.Figure: Matplotlib figure
        """
        assert target_col in df.columns, f"Target column '{target_col}' not found in DataFrame."
        assert failure_id_col in df.columns, f"Failure ID column '{failure_id_col}'"

        failure_data = df[df[target_col] == True].copy()
        
        if len(failure_data) == 0:
            print("No failure data available for heatmap")
            return plt.figure()
        
        # Calculate statistics
        agg_dict = {col: agg_funcs for col in sensor_columns}
        agg_dict[failure_id_col] = 'count'

        failures_summary = failure_data.groupby(failure_id_col).agg(agg_dict)

        # Flatten the multi-level columns
        df_flat = failures_summary.copy()
        df_flat.columns = [f"{col[0]}_{col[1]}" for col in df_flat.columns]
        
        # Separate sensor variables from duration
        sensor_cols = [col for col in df_flat.columns if col != f'{failure_id_col}_count']

        # Create figure with subplots
        fig, axes = plt.subplots(1, 2, figsize=figsize, gridspec_kw={'width_ratios': [4, 1]})
        
        # Plot 1: Sensor variables heatmap (normalized by column)
        sensor_data = df_flat[sensor_cols]
        sensor_normalized = sensor_data.apply(lambda x: (x - x.min()) / (x.max() - x.min()), axis=0)
        
        sns.heatmap(sensor_normalized, 
                    annot=sensor_data.round(1),
                    cmap='RdYlBu_r', 
                    cbar_kws={'label': 'Normalized Scale (0-1)'}, 
                    ax=axes[0],
                    fmt='g')
        
        axes[0].set_title('Sensor Variables (Each Column Independently Scaled)')
        axes[0].set_ylabel(failure_id_col)
        
        # Plot 2: Duration heatmap
        duration_data = df_flat[[f'{failure_id_col}_count']]
        duration_normalized = (duration_data - duration_data.min()) / (duration_data.max() - duration_data.min())
        
        sns.heatmap(duration_normalized,
                    annot=duration_data,
                    cmap='Reds',
                    cbar_kws={'label': 'Duration Scale'},
                    ax=axes[1],
                    fmt='g')
        
        axes[1].set_title('Duration\n(Cycles)')
        axes[1].set_ylabel('')
        
        plt.suptitle('Failure Events - Sensor Statistics Heatmap', fontsize=14, y=1.02)
        plt.tight_layout()
        return fig
    
    def plot_preset_configurations_per_failure(self, df: pd.DataFrame,
                                              target_col: str = 'Fail',
                                              preset_col: str = 'Preset_1',
                                              figsize: Tuple[int, int] = (10, 5)) -> plt.Figure:
        """
        Create a stacked bar plot showing preset configurations per failure event.
        
        Args:
            df (pd.DataFrame): DataFrame with failure events
            preset_col (str): Preset column to analyze
            figsize (Tuple[int, int]): Figure size
            
        Returns:
            plt.Figure: Matplotlib figure
        """
        failure_data = df[df[target_col] == True].copy()
        
        if len(failure_data) == 0:
            print("No failure data available for preset analysis")
            return plt.figure()
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create stacked barplot
        failure_data.groupby(['Fail_index', preset_col]).size().unstack().plot(
            kind='bar',
            stacked=True,
            colormap='Paired',
            ax=ax
        )
        
        # Adjust axes and labels
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_ylabel('Number of cycles')
        ax.set_xlabel('Failure event id')
        ax.set_title(f'{preset_col} Configurations per Failure Event')
        
        plt.tight_layout()
        return fig