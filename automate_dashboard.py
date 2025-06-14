import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from typing import Dict, List, Optional, Any, TypedDict
from dataclasses import dataclass
from enum import Enum
import json
import chardet
from langgraph.graph import Graph, END
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
import os 

# Remove SSL certificate file if it exists
os.environ.pop("SSL_CERT_FILE", None)

# Configure Streamlit
st.set_page_config(
    page_title="Smart Dashboard Generator",
    layout="wide",
    initial_sidebar_state="expanded"
)

# TypedDict definitions
class DataInsightDict(TypedDict):
    column_name: str
    data_type: str
    unique_values: int
    null_percentage: float
    distribution_type: str
    correlation_strength: float
    seasonal_pattern: bool
    outliers_present: bool

class ChartDict(TypedDict):
    figure: go.Figure
    title: str
    reasoning: str
    priority: int

class DataAnalysisStateDict(TypedDict):
    data: Optional[pd.DataFrame]
    data_insights: List[DataInsightDict]
    dashboard_plan: Optional['DashboardPlan']
    charts: List[ChartDict]
    error_message: Optional[str]

# LLM Configuration
@st.cache_resource
def get_llm() -> ChatGroq:
    """Cache LLM instance to avoid recreation"""
    try:
        api_key = st.secrets.get("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not found in secrets")
        
        return ChatGroq(
            model_name="llama3-70b-8192",
            temperature=0.1, 
            api_key=api_key,
            max_retries=2,
            request_timeout=30
        )
    except Exception as e:
        st.error(f"Error initializing LLM: {str(e)}")
        return None

# Data Models
class ChartType(Enum):
    LINE = "line"
    BAR = "bar"
    SCATTER = "scatter"
    PIE = "pie"
    AREA = "area"
    HISTOGRAM = "histogram"
    BOX = "box"

class ChartRecommendation(BaseModel):
    chart_type: str = Field(description="Type of chart to create (line, bar, scatter, pie, histogram, heatmap, box)")
    title: str = Field(description="Chart title")
    x_column: str = Field(description="X-axis column", default="")
    y_column: str = Field(description="Y-axis column", default="")
    color_column: str = Field(description="Color grouping column (optional)", default="")
    size_column: str = Field(description="Size column for scatter plots (optional)", default="")
    aggregation: str = Field(description="Aggregation method if needed", default="")
    reasoning: str = Field(description="Why this chart is recommended")
    priority: int = Field(description="Priority score 1-10")

class DashboardPlan(BaseModel):
    charts: List[ChartRecommendation] = Field(description="List of recommended charts")
    data_summary: str = Field(description="Summary of the data characteristics")
    key_insights: List[str] = Field(description="Key insights from the data")

# Helper functions
def create_initial_state() -> DataAnalysisStateDict:
    """Create initial state dictionary"""
    return DataAnalysisStateDict(
        data=None,
        data_insights=[],
        dashboard_plan=None,
        charts=[],
        error_message=None
    )

def create_data_insight(
    column_name: str,
    data_type: str,
    unique_values: int,
    null_percentage: float,
    distribution_type: str,
    correlation_strength: float,
    seasonal_pattern: bool,
    outliers_present: bool
) -> DataInsightDict:
    """Create a data insight dictionary"""
    return DataInsightDict(
        column_name=column_name,
        data_type=data_type,
        unique_values=unique_values,
        null_percentage=null_percentage,
        distribution_type=distribution_type,
        correlation_strength=correlation_strength,
        seasonal_pattern=seasonal_pattern,
        outliers_present=outliers_present
    )

def create_chart_dict(figure: go.Figure, title: str, reasoning: str, priority: int) -> ChartDict:
    """Create a chart dictionary"""
    return ChartDict(
        figure=figure,
        title=title,
        reasoning=reasoning,
        priority=priority
    )

# LangGraph Node Functions
def analyze_data_characteristics(state: DataAnalysisStateDict) -> DataAnalysisStateDict:
    """Analyze uploaded data to understand its characteristics"""
    if state["data"] is None:
        state["error_message"] = "No data provided"
        return state
    
    df = state["data"].copy()
    # Clean column names
    df.columns = df.columns.str.replace(' ', '_').str.replace('[^a-zA-Z0-9_]', '', regex=True)
    state["data"] = df
    
    insights = []
    
    for col in df.columns:
        try:
            # Basic statistics
            unique_vals = df[col].nunique()
            null_pct = (df[col].isnull().sum() / len(df)) * 100
            
            # Data type analysis
            if pd.api.types.is_numeric_dtype(df[col]):
                data_type = "numeric"
                # Check for skewness
                try:
                    skewness = df[col].skew()
                    distribution = "normal" if abs(skewness) < 1 else "skewed"
                except:
                    distribution = "unknown"
                
                # Check for outliers using IQR
                try:
                    Q1 = df[col].quantile(0.25)
                    Q3 = df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    outliers = len(df[(df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)]) > 0
                except:
                    outliers = False
                    
            elif pd.api.types.is_datetime64_any_dtype(df[col]):
                data_type = "datetime"
                distribution = "temporal"
                outliers = False
            else:
                data_type = "categorical"
                distribution = "categorical"
                outliers = False
            
            # Correlation with other numeric columns
            try:
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 1 and col in numeric_cols:
                    corr_matrix = df[numeric_cols].corr()
                    max_corr = corr_matrix[col].abs().nlargest(2).iloc[1] if len(corr_matrix[col]) > 1 else 0
                else:
                    max_corr = 0
            except:
                max_corr = 0
            
            # Seasonal pattern detection (basic)
            seasonal = data_type == "datetime" or (data_type == "numeric" and unique_vals > 10)
            
            insight = create_data_insight(
                column_name=col,
                data_type=data_type,
                unique_values=unique_vals,
                null_percentage=null_pct,
                distribution_type=distribution,
                correlation_strength=float(max_corr),
                seasonal_pattern=seasonal,
                outliers_present=outliers
            )
            insights.append(insight)
            
        except Exception as e:
            st.warning(f"Error analyzing column {col}: {str(e)}")
            continue
    
    state["data_insights"] = insights
    return state

def generate_dashboard_plan(state: DataAnalysisStateDict) -> DataAnalysisStateDict:
    """Use LLM to generate intelligent dashboard plan"""
    
    groq_llm = get_llm()
    if groq_llm is None:
        state["error_message"] = "LLM not available"
        return state
    
    # Prepare data summary for LLM
    data_summary = {
        "shape": state["data"].shape,
        "columns": []
    }
    
    for insight in state["data_insights"]:
        col_info = {
            "name": insight["column_name"],
            "type": insight["data_type"],
            "unique_values": insight["unique_values"],
            "null_percentage": insight["null_percentage"],
            "has_outliers": insight["outliers_present"],
            "correlation_strength": insight["correlation_strength"]
        }
        data_summary["columns"].append(col_info)
    
    # Create prompt for LLM
    parser = PydanticOutputParser(pydantic_object=DashboardPlan)
    
    prompt_template = ChatPromptTemplate.from_template("""
    You are an expert data analyst and dashboard designer. Analyze the following dataset and create a comprehensive dashboard plan.

    Dataset Information:
    - Shape: {rows} rows, {cols} columns
    - Columns: {column_info}

    Requirements:
    1. Create exactly 4-6 different chart recommendations
    2. Use ONLY these chart types: line, bar, scatter, pie, histogram,box
    3. Select the most suitable columns for each chart type
    4. Prioritize charts that reveal key insights
    5. Consider data types, distributions, and relationships
    6. Avoid redundant visualizations

    IMPORTANT FORMATTING RULES:
    - chart_type must be exactly one of: line, bar, scatter, pie, histogram,box
    - x_column and y_column must be valid column names from the dataset or empty string ""
    - For box plots: use y_column for the numeric variable, x_column can be empty or categorical
    - For histograms: use x_column for the variable, y_column should be empty
    - For pie charts: use x_column for categories, y_column for values or empty for counts
    

    Chart Type Guidelines:
    - line: Time series data, trends over time (requires x_column and y_column)
    - bar: Categorical comparisons, rankings (requires x_column, y_column optional)
    - scatter: Relationships between numeric variables (requires x_column and y_column)
    - pie: Proportions of categorical data (requires x_column, y_column optional)
    - histogram: Distribution of single numeric variable (requires x_column)
    - box: Distribution analysis, outlier detection (requires y_column, x_column optional)

    {format_instructions}

    Provide a comprehensive dashboard plan with diverse, insightful visualizations.
    """)
    
    # Format column information
    column_info = json.dumps(data_summary["columns"], indent=2)
    
    # Generate plan using LLM
    try:
        messages = prompt_template.format_messages(
            rows=data_summary["shape"][0],
            cols=data_summary["shape"][1],
            column_info=column_info,
            format_instructions=parser.get_format_instructions()
        )
        
        response = groq_llm.invoke(messages)
        dashboard_plan = parser.parse(response.content)
        
        # Clean and validate charts
        valid_charts = []
        for chart in dashboard_plan.charts:
            # Normalize chart type
            chart.chart_type = normalize_chart_type(chart.chart_type)
            
            # Ensure required fields are not None
            if not chart.x_column:
                chart.x_column = ""
            if not chart.y_column:
                chart.y_column = ""
            if not chart.color_column:
                chart.color_column = ""
            if not chart.size_column:
                chart.size_column = ""
            if not chart.aggregation:
                chart.aggregation = ""
            
            # Validate chart
            if validate_chart_recommendation(chart, state["data"]):
                valid_charts.append(chart)
        
        dashboard_plan.charts = valid_charts[:6]  
        state["dashboard_plan"] = dashboard_plan
        
    except Exception as e:
        # st.error(f"Error generating dashboard plan: {str(e)}")
        # Fallback to basic charts
        state["dashboard_plan"] = create_fallback_dashboard_plan(state)
    
    return state

def normalize_chart_type(chart_type: str) -> str:
    """Normalize chart type to expected format"""
    if not chart_type:
        return "bar"
    
    chart_type = chart_type.lower().strip()
    
    # Handle common variations
    type_mapping = {
        "line chart": "line",
        "line plot": "line",
        "bar chart": "bar",
        "bar plot": "bar",
        "scatter plot": "scatter",
        "scatterplot": "scatter",
        "pie chart": "pie",
        "histogram": "histogram",
        "box plot": "box",
        "boxplot": "box"
    }
    
    return type_mapping.get(chart_type, chart_type)

def validate_chart_recommendation(chart: ChartRecommendation, df: pd.DataFrame) -> bool:
    """Validate if chart recommendation is feasible with the data"""
    try:
        # Check if columns exist (empty string is allowed)
        if chart.x_column and chart.x_column not in df.columns:
            return False
        if chart.y_column and chart.y_column not in df.columns:
            return False
        if chart.color_column and chart.color_column not in df.columns:
            chart.color_column = ""
        if chart.size_column and chart.size_column not in df.columns:
            chart.size_column = ""
        
        # Validate chart type requirements
        if chart.chart_type == "scatter":
            if not chart.x_column or not chart.y_column:
                return False
            if not pd.api.types.is_numeric_dtype(df[chart.x_column]) or not pd.api.types.is_numeric_dtype(df[chart.y_column]):
                return False
        
        elif chart.chart_type == "line":
            if not chart.x_column or not chart.y_column:
                return False
            if not pd.api.types.is_numeric_dtype(df[chart.y_column]):
                return False
        
        elif chart.chart_type == "pie":
            if not chart.x_column:
                return False
            if df[chart.x_column].nunique() > 15:
                return False
        
        elif chart.chart_type == "bar":
            if not chart.x_column:
                return False
                
        elif chart.chart_type == "histogram":
            if not chart.x_column:
                return False
            if not pd.api.types.is_numeric_dtype(df[chart.x_column]):
                return False
        
        return True
    except Exception as e:
        st.warning(f"Validation error for chart {chart.title}: {str(e)}")
        return False

def create_fallback_dashboard_plan(state: DataAnalysisStateDict) -> DashboardPlan:
    """Create basic dashboard plan if LLM fails"""
    df = state["data"]
    charts = []
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
    
    # Basic charts
    if len(numeric_cols) >= 2:
        charts.append(ChartRecommendation(
            chart_type="scatter",
            title=f"{numeric_cols[0]} vs {numeric_cols[1]}",
            x_column=numeric_cols[0],
            y_column=numeric_cols[1],
            reasoning="Scatter plot to show relationship between numeric variables",
            priority=8
        ))
    
    if len(categorical_cols) >= 1:
        charts.append(ChartRecommendation(
            chart_type="bar",
            title=f"Count by {categorical_cols[0]}",
            x_column=categorical_cols[0],
            y_column="",
            aggregation="count",
            reasoning="Bar chart for categorical analysis",
            priority=6
        ))
    
    if len(numeric_cols) >= 1:
        charts.append(ChartRecommendation(
            chart_type="histogram",
            title=f"Distribution of {numeric_cols[0]}",
            x_column=numeric_cols[0],
            y_column="",
            reasoning="Histogram to show distribution of numeric variable",
            priority=7
        ))
    
    return DashboardPlan(
        charts=charts,
        data_summary="Fallback dashboard plan created due to LLM unavailability",
        key_insights=["Basic analysis of uploaded data", "Showing fundamental data patterns"]
    )

def create_visualizations(state: DataAnalysisStateDict) -> DataAnalysisStateDict:
    """Create actual visualizations based on the dashboard plan"""
    if not state["dashboard_plan"]:
        state["error_message"] = "No dashboard plan available"
        return state
    
    charts = []
    df = state["data"]
    
    for chart_rec in state["dashboard_plan"].charts:
        try:
            chart_data = create_single_chart(df, chart_rec)
            if chart_data:
                charts.append(chart_data)
        except Exception as e:
            # st.warning(f"Error creating chart '{chart_rec.title}': {str(e)}")
            continue
    
    state["charts"] = charts
    return state

def create_single_chart(df: pd.DataFrame, chart_rec: ChartRecommendation) -> Optional[ChartDict]:
    """Create a single chart based on recommendation"""
    try:
        chart_type = chart_rec.chart_type.lower()
        
        if chart_type == "scatter":
            fig = px.scatter(
                df, 
                x=chart_rec.x_column, 
                y=chart_rec.y_column,
                color=chart_rec.color_column if chart_rec.color_column else None,
                size=chart_rec.size_column if chart_rec.size_column else None,
                title=chart_rec.title,
                template="plotly_white"
            )
        
        elif chart_type == "line":
            fig = px.line(
                df, 
                x=chart_rec.x_column, 
                y=chart_rec.y_column,
                color=chart_rec.color_column if chart_rec.color_column else None,
                title=chart_rec.title,
                template="plotly_white"
            )
        
        elif chart_type == "bar":
            if chart_rec.aggregation == "count" or not chart_rec.y_column:
                fig = px.histogram(
                    df, 
                    x=chart_rec.x_column,
                    title=chart_rec.title,
                    template="plotly_white"
                )
            else:
                # Group and aggregate data for bar chart
                try:
                    if chart_rec.aggregation == "sum":
                        agg_data = df.groupby(chart_rec.x_column)[chart_rec.y_column].sum().reset_index()
                    elif chart_rec.aggregation == "mean":
                        agg_data = df.groupby(chart_rec.x_column)[chart_rec.y_column].mean().reset_index()
                    else:
                        agg_data = df.groupby(chart_rec.x_column)[chart_rec.y_column].sum().reset_index()
                    
                    # Limit to top 20 categories for readability
                    agg_data = agg_data.nlargest(20, chart_rec.y_column)
                    
                    fig = px.bar(
                        agg_data, 
                        x=chart_rec.x_column, 
                        y=chart_rec.y_column,
                        title=chart_rec.title,
                        template="plotly_white"
                    )
                except Exception as e:
                    st.warning(f"Error creating aggregated bar chart: {str(e)}")
                    return None
        
        elif chart_type == "pie":
            try:
                if chart_rec.y_column and chart_rec.y_column in df.columns:
                    # Use y_column for values
                    agg_data = df.groupby(chart_rec.x_column)[chart_rec.y_column].sum().head(10)
                else:
                    # Use counts
                    agg_data = df[chart_rec.x_column].value_counts().head(10)
                
                fig = px.pie(
                    values=agg_data.values,
                    names=agg_data.index,
                    title=chart_rec.title,
                    template="plotly_white"
                )
            except Exception as e:
                st.warning(f"Error creating pie chart: {str(e)}")
                return None
        
        elif chart_type == "histogram":
            fig = px.histogram(
                df,
                x=chart_rec.x_column,
                title=chart_rec.title,
                template="plotly_white",
                nbins=30
            )
        
        elif chart_type == "box":
            fig = px.box(
                df,
                y=chart_rec.y_column,
                x=chart_rec.x_column if chart_rec.x_column else None,
                title=chart_rec.title,
                template="plotly_white"
            )
        
        else:
            st.warning(f"Unknown chart type: {chart_type}")
            return None
        
        # Update layout
        fig.update_layout(
            height=400,
            showlegend=True,
            font=dict(size=12),
            title_font_size=16,
            margin=dict(l=50, r=50, t=70, b=50)
        )
        
        return create_chart_dict(
            figure=fig,
            title=chart_rec.title,
            reasoning=chart_rec.reasoning,
            priority=chart_rec.priority
        )
        
    except Exception as e:
        st.error(f"Error creating chart '{chart_rec.title}': {str(e)}")
        return None

# LangGraph Workflow
def create_dashboard_workflow():
    """Create the LangGraph workflow for dashboard generation"""
    
    workflow = Graph()
    
    # Add nodes
    workflow.add_node("analyze_data", analyze_data_characteristics)
    workflow.add_node("generate_plan", generate_dashboard_plan)
    workflow.add_node("create_charts", create_visualizations)
    
    # Add edges
    workflow.add_edge("analyze_data", "generate_plan")
    workflow.add_edge("generate_plan", "create_charts")
    workflow.add_edge("create_charts", END)
    
    # Set entry point
    workflow.set_entry_point("analyze_data")
    
    return workflow.compile()

# File loading functions
def load_data_file(uploaded_file) -> Optional[pd.DataFrame]:
    """Load data from uploaded file with proper error handling"""
    try:
        # Detect encoding
        uploaded_file.seek(0)
        raw_bytes = uploaded_file.read(10000)
        result = chardet.detect(raw_bytes)
        encoding = result.get('encoding', 'utf-8')
        
        # Reset file pointer
        uploaded_file.seek(0)
        
        # Load based on file extension
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file, encoding=encoding)
        elif uploaded_file.name.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(uploaded_file)
        else:
            st.error("Unsupported file format")
            return None
        
        # Basic data validation
        if df.empty:
            st.error("The uploaded file is empty")
            return None
        
        if len(df.columns) == 0:
            st.error("No columns found in the data")
            return None
        
        return df
        
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        return None

# Streamlit UI
def main():
    st.title("Smart Automated Dashboard Generator")
    st.markdown("Upload your data and get intelligent, automated visualizations powered by LangGraph and AI")
    
    # Sidebar
    with st.sidebar:
        st.header("📁 Data Upload")
        uploaded_file = st.file_uploader(
            "Choose a data file",
            type=['csv', 'xlsx', 'xls'],
            help="Upload CSV, Excel, or JSON files"
        )
        
        if uploaded_file:
            st.success("✅ File uploaded successfully!")
            
            # File info
            st.info(f"**File:** {uploaded_file.name}")
            st.info(f"**Size:** {uploaded_file.size:,} bytes")
    
    if uploaded_file is not None:
        # Load data
        with st.spinner("Loading data..."):
            df = load_data_file(uploaded_file)
        
        if df is not None:
            # Data preview
            st.header("📊 Data Preview")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Rows", f"{len(df):,}")
            with col2:
                st.metric("Columns", len(df.columns))
            with col3:
                st.metric("Missing Values", f"{df.isnull().sum().sum():,}")
            
            with st.expander("View Raw Data", expanded=False):
                st.dataframe(df.head(100), use_container_width=True)
            
            # Generate dashboard
            if st.button("🚀 Generate Smart Dashboard", type="primary", use_container_width=True):
                
                with st.spinner("🤖 AI is analyzing your data and creating intelligent visualizations..."):
                    
                    # Initialize state
                    state = create_initial_state()
                    state["data"] = df
                    
                    # Create and run workflow
                    workflow = create_dashboard_workflow()
                    
                    try:
                        # Run the workflow
                        result = workflow.invoke(state)
                        
                        if result["error_message"]:
                            st.error(f"Error: {result['error_message']}")
                            return
                        
                        # Display results
                        st.header("🎯 AI-Generated Dashboard")
                        
                        # Key insights
                        if result["dashboard_plan"] and result["dashboard_plan"].key_insights:
                            st.subheader("💡 Key Insights")
                            for insight in result["dashboard_plan"].key_insights:
                                st.info(f"• {insight}")
                        
                        # Display charts
                        if result["charts"]:
                            st.subheader("📈 Intelligent Visualizations")
                            
                            # Sort charts by priority
                            sorted_charts = sorted(result["charts"], key=lambda x: x['priority'], reverse=True)
                            
                            # Display charts
                            for i, chart_data in enumerate(sorted_charts):
                                st.plotly_chart(
                                    chart_data['figure'], 
                                    use_container_width=True,
                                    key=f"chart_{i}"
                                )
                                
                                with st.expander(f"💭 Why this chart? - {chart_data['title']}", expanded=False):
                                    st.write(chart_data['reasoning'])
                        
                        else:
                            st.warning("No charts were generated. Please check your data format.")
                    
                    except Exception as e:
                        st.error(f"An error occurred during dashboard generation: {str(e)}")
                        st.error("Please try with a different dataset or contact support.")
                        st.exception(e) 
    
    else:
        # Welcome message
        st.markdown("""
        **Supported formats:** CSV, Excel (XLSX/XLS)
        """)

if __name__ == "__main__":
    main()



