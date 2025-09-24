import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
import seaborn as sns

def show_boxplots(df: pd.DataFrame, layout: str = "separate"):
    """
    Displays boxplots for all numeric columns in the DataFrame.

    Args:
        df (pd.DataFrame): Input DataFrame.
        layout (str): 'separate' = one plot per column,
                      'grid' = all plots in a grid.
    """
    numeric_cols = df.select_dtypes(include="number").columns

    if layout == "grid":
        n = len(numeric_cols)
        rows = (n + 2) // 3
        fig, axes = plt.subplots(rows, 3, figsize=(15, 5 * rows))
        axes = axes.flatten()

        for i, col in enumerate(numeric_cols):
            df.boxplot(column=col, ax=axes[i])
            axes[i].set_title(f'Boxplot for {col}')
            axes[i].set_ylabel(col)

        for j in range(i + 1, len(axes)):
            axes[j].axis("off")

        plt.tight_layout()
        plt.show()
    else:
        for column in numeric_cols:
            plt.figure(figsize=(6, 4))
            df.boxplot(column=column)
            plt.title(f'Boxplot for {column}')
            plt.ylabel(column)
            plt.tight_layout()
            plt.show()


def show_histograms(df: pd.DataFrame, bins: int = 10, layout: str = "separate", bell_curve: bool = False):
    """
    Displays histograms for all numeric columns in the DataFrame.
    Optionally overlays a normal distribution (bell curve) if bell_curve=True.

    Args:
        df (pd.DataFrame): Input DataFrame.
        bins (int): Number of bins for histograms. Defaults to 10.
        layout (str): 'separate' = one plot per column,
                      'grid' = all plots in a grid.
        bell_curve (bool): If True, overlays a normal distribution curve.
    """
    numeric_cols = df.select_dtypes(include="number").columns

    if layout == "grid":
        n = len(numeric_cols)
        rows = (n + 2) // 3
        fig, axes = plt.subplots(rows, 3, figsize=(15, 5 * rows))
        axes = axes.flatten()

        for i, col in enumerate(numeric_cols):
            data_df = df[col].dropna()
            axes[i].hist(data_df, bins=bins, density=True, alpha=0.7, color='tab:blue', edgecolor='black')
            if bell_curve:
                mu, std = data_df.mean(), data_df.std()
                xmin, xmax = axes[i].get_xlim()
                x = np.linspace(xmin, xmax, 100)
                p = (1/(std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / std) ** 2)
                axes[i].plot(x, p, 'r', linewidth=2)
            axes[i].set_title(f'Histogram of {col}')
            axes[i].set_xlabel(col)
            axes[i].set_ylabel('Frequency')
            axes[i].grid(True)

        for j in range(i + 1, len(axes)):
            axes[j].axis("off")

        plt.tight_layout()
        plt.show()
    else:
        for col in numeric_cols:
            plt.figure(figsize=(6, 4))
            data_df = df[col].dropna()
            plt.hist(data_df, bins=bins, density=True, alpha=0.7, color='tab:blue', edgecolor='black')
            if bell_curve:
                mu, std = data_df.mean(), data_df.std()
                xmin, xmax = plt.xlim()
                x = np.linspace(xmin, xmax, 100)
                p = (1/(std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / std) ** 2)
                plt.plot(x, p, 'r', linewidth=2)
            plt.title(f'Histogram of {col}')
            plt.xlabel(col)
            plt.ylabel("Frequency")
            plt.grid(True)
            plt.tight_layout()
            plt.show()


def show_scatter_matrix(df: pd.DataFrame, figsize: tuple = (12, 12), diagonal: str = "hist"):
    """
    Displays a scatter matrix (pairplot) for numeric columns.

    Args:
        df (pd.DataFrame): Input DataFrame.
        figsize (tuple): Size of the figure. Defaults to (12, 12).
        diagonal (str): Plot type on the diagonal ('hist' or 'kde'). Defaults to 'hist'.
    """
    numeric_cols = df.select_dtypes(include="number").columns
    if len(numeric_cols) < 2:
        print("Scatter matrix requires at least two numeric columns.")
        return
    scatter_matrix(df[numeric_cols], figsize=figsize, diagonal=diagonal)
    plt.suptitle("Scatter Matrix of Numeric Features")
    plt.show()

def show_correlation_heatmap(df: pd.DataFrame):
    """
    Displays a heatmap of the correlation matrix for numeric columns.

    Args:
        df (pd.DataFrame): Input DataFrame.
    """
    numeric_cols = df.select_dtypes(include="number").columns
    if len(numeric_cols) < 2:
        print("Correlation heatmap requires at least two numeric columns.")
        return
    corr = df[numeric_cols].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Matrix of Numeric Features')
    plt.show()


def show_grouped_histograms(
    df: pd.DataFrame,
    bins: int = 10,
    layout: str = "separate",
    category_col: str = "type",
    bell_curve: bool = True,
    max_cols: int = 3,
    alpha: float = 0.7,
):
    """
    Draw side-by-side histograms per category (default 'type') for all numeric columns.
    Overlays a separate normal bell curve per category when bell_curve=True.

    Args:
        df: Input DataFrame.
        bins: Number of bins.
        layout: 'separate' (one figure per column) or 'grid' (subplot grid).
        category_col: Categorical column to split by (default 'type').
        bell_curve: Overlay per-category normal curves if True.
        max_cols: Max columns per row in grid layout.
        alpha: Bar transparency.
    """
    if category_col not in df.columns:
        raise ValueError(f"Category column '{category_col}' not found in DataFrame.")

    numeric_cols = [c for c in df.select_dtypes(include="number").columns if c != category_col]
    if not numeric_cols:
        print("No numeric columns to plot.")
        return

    cats = list(df[category_col].dropna().unique())
    if len(cats) == 0:
        print(f"No non-null categories found in '{category_col}'.")
        return

    # Color mapping per category
    default_colors = plt.rcParams.get("axes.prop_cycle", None)
    palette = (default_colors.by_key()["color"] if default_colors else ["C0", "C1", "C2", "C3"])
    color_map = {cat: palette[i % len(palette)] for i, cat in enumerate(cats)}

    def _plot_column(ax, col: str):
        data_col = df[col].dropna().values
        if data_col.size == 0:
            ax.set_visible(False)
            return

        # Shared bin edges across categories
        bin_edges = np.histogram_bin_edges(data_col, bins=bins)
        k = max(1, len(cats))
        # Width for each category inside each bin (handles variable-width bins)
        bin_widths = np.diff(bin_edges)
        width_each = bin_widths / k

        handles = []
        labels = []

        for j, cat in enumerate(cats):
            vals = df.loc[df[category_col] == cat, col].dropna().values
            if vals.size == 0:
                continue

            counts, _ = np.histogram(vals, bins=bin_edges, density=True)  # density=True to match bell curves
            lefts = bin_edges[:-1] + j * width_each
            bar = ax.bar(
                lefts,
                counts,
                width=width_each,
                align="edge",
                alpha=alpha,
                edgecolor="black",
                color=color_map[cat],
                label=str(cat),
            )
            # Keep one handle per category for legend
            handles.append(bar)
            labels.append(str(cat))

            if bell_curve:
                mu = np.mean(vals)
                std = np.std(vals, ddof=1)
                if std > 0 and np.isfinite(std):
                    x = np.linspace(bin_edges[0], bin_edges[-1], 200)
                    pdf = (1.0 / (std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / std) ** 2)
                    ax.plot(x, pdf, color=color_map[cat], linewidth=2, alpha=0.9)

        ax.set_title(f'Histogram of {col} by {category_col}')
        ax.set_xlabel(col)
        ax.set_ylabel('Density')
        ax.grid(True, alpha=0.25)
        if handles:
            # Use category bars as legend entries
            ax.legend([h for h in handles], labels, title=category_col)

    if layout == "grid":
        n = len(numeric_cols)
        cols = max(1, min(max_cols, n))
        rows = int(np.ceil(n / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4.5 * rows), squeeze=False)
        flat_axes = axes.flatten()

        for ax, col in zip(flat_axes, numeric_cols):
            _plot_column(ax, col)

        # Hide any extra axes
        for i in range(len(numeric_cols), len(flat_axes)):
            fig.delaxes(flat_axes[i])

        plt.tight_layout()
        plt.show()
    else:
        for col in numeric_cols:
            fig, ax = plt.subplots(figsize=(6.5, 4.5))
            _plot_column(ax, col)
            plt.tight_layout()
            plt.show()

def show_binned_data(
    df: pd.DataFrame,
    bins: int = 5,
    column_to_bin: str = 'pH',
    column_to_plot: str = 'density',
    aggregation_method: str = 'mean'
):
    """
    Plots aggregated values (mean or max) of `column_to_plot` grouped by bins of `column_to_bin`.

    Args:
        df: Input DataFrame.
        bins: Number of bins for binning.
        column_to_bin: Column to bin.
        column_to_plot: Column to aggregate and plot.
        aggregation_method: 'mean' or 'max' aggregation.
    """
    if column_to_bin not in df.columns or column_to_plot not in df.columns:
        raise ValueError("Specified columns must be present in the DataFrame.")

    binned_df = df[[column_to_bin, column_to_plot]].copy()
    binned_df['bin'] = pd.cut(binned_df[column_to_bin], bins=bins)

    if aggregation_method == 'max':
        binned_vals = binned_df.groupby('bin')[column_to_plot].max()
        agg_label = 'Max'
    else:
        binned_vals = binned_df.groupby('bin')[column_to_plot].mean()
        agg_label = 'Mean'

    plt.figure(figsize=(10, 6))
    binned_vals.plot(kind='bar', color='skyblue')
    plt.title(f'{agg_label} {column_to_plot} by {column_to_bin} Bin')
    plt.xlabel(f'{column_to_bin} Bin')
    plt.ylabel(f'{agg_label} {column_to_plot}')
    plt.xticks(rotation=45)
    plt.grid(axis='y')
    plt.tight_layout()
    plt.show()
    return binned_vals



