Smart Automated Dashboard Generator AI Tool

An intelligent, AI-powered dashboard generator that automatically analyzes your data and creates meaningful visualizations using LangGraph workflow orchestration and large language models.

Features

- **AI-Powered Analysis**: Leverages LLM to intelligently analyze data characteristics and relationships
- **Automatic Chart Selection**: Smartly chooses the most appropriate visualization types based on data patterns
- **Workflow Orchestration**: Uses LangGraph to create a structured, multi-step analysis pipeline
- **Interactive Visualizations**: Generates beautiful, interactive charts using Plotly
- **Multiple File Formats**: Supports CSV and Excel (XLSX/XLS) files
- **Intelligent Insights**: Provides AI-generated explanations for each visualization choice
- **One-Click Generation**: Simple upload and generate workflow

Architecture

The application uses a sophisticated workflow pipeline powered by LangGraph:

```
Data Upload → Data Analysis → Dashboard Planning → Visualization Creation
     ↓              ↓               ↓                    ↓
File Processing → AI Analysis → LLM Recommendations → Chart Generation
```

Core Components:

1. **Data Analysis Engine**: Automatically detects data types, distributions, correlations, and patterns
2. **AI Planning Module**: Uses Groq's LLaMA model to generate intelligent dashboard recommendations
3. **Visualization Engine**: Creates interactive charts using Plotly based on AI recommendations
4. **Workflow Orchestration**: LangGraph manages the entire pipeline with error handling and fallbacks

Technology Stack

- **Frontend**: Streamlit
- **AI/LLM**: LangChain + Groq (LLaMA 3-70B)
- **Workflow**: LangGraph
- **Visualizations**: Plotly Express & Graph Objects
- **Data Processing**: Pandas, NumPy
- **File Handling**: Chardet for encoding detection

Prerequisites

- Python 3.8+
- Groq API Key ([Get one here](https://console.groq.com/))
- Required Python packages (see requirements.txt)

Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/smart-dashboard-generator.git
   cd smart-dashboard-generator
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up Streamlit secrets**:
   Create `.streamlit/secrets.toml`:
   ```toml
   GROQ_API_KEY = "your_groq_api_key_here"
   ```

4. **Run the application**:
   ```bash
   streamlit run app.py
   ```

Supported Chart Types

The AI can generate the following visualization types:

- **Line Charts**: Time series data and trends
- **Bar Charts**: Categorical comparisons and rankings
- **Scatter Plots**: Relationships between numeric variables
- **Pie Charts**: Proportional data (automatically limits to top 10 categories)
- **Histograms**: Distribution analysis for numeric data
- **Box Plots**: Distribution analysis and outlier detection

How It Works

1. **Upload Your Data**: Drag and drop CSV or Excel files
2. **Automatic Analysis**: AI analyzes data structure, types, and relationships
3. **Smart Planning**: LLM generates intelligent visualization recommendations
4. **Chart Generation**: Creates 4-6 diverse, meaningful visualizations
5. **Insight Explanation**: Provides reasoning for each chart choice

📁 File Format Support

| Format | Extensions | Notes |
|--------|------------|-------|
| CSV | `.csv` | Automatic encoding detection |
| Excel | `.xlsx`, `.xls` | All sheets supported |


Configuration

LLM Settings
- **Model**: LLaMA 3-70B (via Groq)
- **Temperature**: 0.1 (for consistent results)

Chart Limitations
- Maximum 6 charts per dashboard
- Pie charts limited to top 10 categories
- Bar charts limited to top 20 categories for readability

Error Handling

The application includes comprehensive error handling:

- **File Loading**: Encoding detection and format validation
- **Data Validation**: Empty file and column checks
- **LLM Fallbacks**: Basic chart generation if AI fails
- **Chart Validation**: Ensures compatibility between data and chart types

Acknowledgments

- [LangChain](https://langchain.com/) for LLM orchestration
- [LangGraph](https://langchain-ai.github.io/langgraph/) for workflow management
- [Groq](https://groq.com/) for fast LLM inference
- [Plotly](https://plotly.com/) for interactive visualizations
- [Streamlit](https://streamlit.io/) for the web interface

