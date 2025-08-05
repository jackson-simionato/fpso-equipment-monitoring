"""
Visualization module for data analysis exploration.

This module provides plotting functions using seaborn for exploratory data analysis
of FPSO equipment monitoring data.
"""

import pandas as pd
import matplotlib.pyplot as plt
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