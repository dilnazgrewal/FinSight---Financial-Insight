import plotly.express as px
import pandas as pd
import plotly.graph_objects as go

def plot_expense_by_category(df: pd.DataFrame, top_n: int = 8):
    """
    FINAL VERSION: Plots a professional pie chart with automatic grouping and a high-contrast color scheme.
    """
    cat_sum = df[df['Debit'] > 0].groupby("Category")["Debit"].sum().reset_index()
    if cat_sum.empty:
        return None

    # Automatic Grouping Logic
    if len(cat_sum) > top_n:
        cat_sum = cat_sum.sort_values(by="Debit", ascending=False)
        top_df = cat_sum.head(top_n)
        other_sum = cat_sum.tail(len(cat_sum) - top_n)['Debit'].sum()
        other_row = pd.DataFrame([{'Category': 'Other Expenses', 'Debit': other_sum}])
        plot_df = pd.concat([top_df, other_row], ignore_index=True)
    else:
        plot_df = cat_sum

    # Visualization with a better color palette
    plot_df['Percentage'] = 100 * plot_df['Debit'] / plot_df['Debit'].sum()
    
    fig = px.pie(
        plot_df, 
        names="Category", 
        values="Debit", 
        title=f"Top {top_n} Spending Categories", 
        hole=0.4,
        # UPDATED: Switched to a more vibrant and professional color palette
        color_discrete_sequence=px.colors.qualitative.Antique 
    )

    fig.update_traces(
        text=[f"{p:.1f}%" if p > 4 else '' for p in plot_df['Percentage']],
        textinfo='text',
        textposition='inside',
        hovertemplate="<b>%{label}</b><br>Amount: ₹%{value:,.2f}<br>Share: %{percent}<extra></extra>"
    )
    
    fig.update_layout(
        showlegend=True,
        legend_title_text='Categories',
        legend=dict(
            orientation="v", yanchor="top", y=1, xanchor="left", x=1.05
        )
    )
    return fig

def plot_need_vs_want(df: pd.DataFrame):
    """
    IMPROVED: Plots a bar chart comparing 'Need' vs 'Want' spending with better colors.
    """
    if "Classification" not in df.columns:
        return None
        
    type_sum = df.groupby("Classification")["Debit"].sum().reset_index()
    if type_sum.empty:
        return None

    # UPDATED: Changed orange to a more conventional muted red for 'Want'
    color_map = {'Need': '#1f77b4', 'Want': "#C3CCD3", 'Other': '#7f7f7f'}
    
    fig = px.bar(
        type_sum, 
        x="Classification", 
        y="Debit", 
        title="Need vs. Want Spending", 
        text_auto='.2s',
        color="Classification",
        color_discrete_map=color_map
    )
    fig.update_layout(showlegend=False)
    return fig

def plot_top_expenses(df: pd.DataFrame, top_n: int = 10):
    """
    Plots a horizontal bar chart of the top N largest individual expenses.
    """
    if 'Debit' not in df.columns or df.empty:
        return None
        
    top_spend_df = df[df['Debit'] > 0].nlargest(top_n, 'Debit').copy()
    if top_spend_df.empty:
        return None

    top_spend_df['Description'] = top_spend_df['Description'].fillna('N/A')
    top_spend_df['Label'] = top_spend_df['Description'].str.strip() + " (" + pd.to_datetime(top_spend_df['Date']).dt.strftime('%b %d') + ")"
    
    fig = px.bar(
        top_spend_df, 
        x='Debit', 
        y='Label', 
        title=f'Top {top_n} Largest Expenses', 
        orientation='h',
        text='Debit'
    )
    fig.update_traces(texttemplate='₹%{text:,.0f}', textposition='outside')
    fig.update_layout(yaxis={'categoryorder':'total ascending'})
    return fig

def plot_monthly_trends(df: pd.DataFrame):
    """
    Plots monthly income vs. expenses, switching to a bar chart for single-month data.
    """
    if 'Date' not in df.columns or df.empty:
        return None
        
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df.dropna(subset=['Date'], inplace=True)

    df['Month'] = df['Date'].dt.to_period('M').astype(str)
    monthly = df.groupby('Month')[['Debit', 'Credit']].sum().reset_index()
    
    if monthly.empty:
        return None

    if len(monthly) < 2:
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=['Income'],
            y=[monthly['Credit'].iloc[0]],
            name='Income',
            text=monthly['Credit'],
            texttemplate='₹%{text:,.0f}'
        ))
        fig.add_trace(go.Bar(
            x=['Expense'],
            y=[monthly['Debit'].iloc[0]],
            name='Expense',
            text=monthly['Debit'],
            texttemplate='₹%{text:,.0f}'
        ))
        fig.update_layout(title_text=f"Income vs. Expense for {monthly['Month'].iloc[0]}")
    else:
        fig = px.line(
            monthly, 
            x='Month', 
            y=['Credit', 'Debit'], 
            title='Monthly Income vs. Expense Trends',
            markers=True
        )
        fig.data[0].name = 'Income'
        fig.data[1].name = 'Expense'

    return fig