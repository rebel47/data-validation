import streamlit as st
import pandas as pd
import google.generativeai as genai
from dotenv import load_dotenv
import json
import re
from collections import defaultdict
import plotly.graph_objects as go
from datetime import datetime
import numpy as np
import os

# Load environment variables
load_dotenv(".env")

# Initialize Gemini client
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel('gemini-2.5-flash')

# Configuration
CONFIG = {
    'CURRENT_USER': "Ayaz",
    'ALLOWED_DATA_TYPES': ['string', 'integer', 'float', 'date', 'boolean', 'category'],
    'VALIDATION_MAPPINGS': {
        'data_type': 'Type Match',
        'range': 'Out of Range', 
        'duplicate': 'Has Duplicates',
        'null': 'Has Nulls',
        'format': 'Has Format Issues'
    }
}


# ----- SIMPLE DATA MODELS -----

class ValidationRule:
    """Simplified validation rule class"""
    def __init__(self, column, data_type, constraints=None):
        self.column = column
        self.data_type = data_type if data_type in CONFIG['ALLOWED_DATA_TYPES'] else 'string'
        self.constraints = constraints or []

class ValidationRules:
    """Container for validation rules"""
    def __init__(self, rules):
        self.rules = rules

# ----- DATA LOADING AND ANALYSIS FUNCTIONS -----

def load_data(uploaded_file):
    """Load dataset from uploaded file with improved error handling."""
    file_name = uploaded_file.name
    file_extension = file_name.lower().split('.')[-1]
    
    try:
        if file_extension == 'csv':
            return pd.read_csv(uploaded_file)
        elif file_extension == 'json':
            return pd.read_json(uploaded_file)
        elif file_extension in ['xlsx', 'xls']:
            return pd.read_excel(uploaded_file)
        else:
            raise ValueError(f"Unsupported file format: .{file_extension}. Please use CSV, JSON, or Excel files.")
    except Exception as e:
        raise ValueError(f"Error reading file: {str(e)}")

def profile_data(df):
    """Extract schema and basic stats."""
    schema = {}
    for col in df.columns:
        schema[col] = {
            "detected_type": str(df[col].dtype),
            "missing_values": df[col].isnull().sum(),
            "unique_values": df[col].nunique(),
            "sample_values": df[col].dropna().sample(min(25, len(df))).tolist()
        }
    return schema

# ----- SCHEMA SUGGESTION AND VALIDATION -----

def generate_schema_suggestions(schema):
    """Ask Gemini to return structured validation rules as JSON with type comparison."""
    prompt = f"""
        You are a data quality expert. Given this dataset schema, return validation rules as JSON, including:
        - `column`: Column name
        - `data_type`: Expected type (string, integer, float, date, boolean, category)
        - `actual_type`: Current data type detected in the dataset
        - `type_match`: Boolean indicating if expected type matches actual type
        - `type_conversion_needed`: Boolean indicating if type conversion is recommended
        - `conversion_recommendation`: String with recommendation for type conversion (if needed)
        - `constraints`: List of constraints (e.g., unique, non-null, regex pattern)
        - `example_values`: 5 Sample valid values

        Schema:
        {schema}

        Instructions:
            - Return only valid JSON code.
            - Do NOT include explanations, comments, markdown, or extra text.
            - Make sure the constraints are applicable to the data type.
            - For string columns, consider regex patterns when appropriate.
            - For numeric columns, include appropriate min/max values.
            - Compare actual type with what it SHOULD be based on the data.
            - Provide specific recommendations for data type conversions.

        Example Output:
            {{
                "rules": [
                    {{
                        "column": "email", 
                        "data_type": "string", 
                        "actual_type": "object", 
                        "type_match": true, 
                        "type_conversion_needed": false,
                        "conversion_recommendation": "",
                        "constraints": ["non-null", "regex: ^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\\.[a-zA-Z0-9-.]+$"], 
                        "example_values": ["user@example.com"]
                    }},
                    {{
                        "column": "age", 
                        "data_type": "integer", 
                        "actual_type": "float64", 
                        "type_match": false, 
                        "type_conversion_needed": true,
                        "conversion_recommendation": "Convert to integer using df['age'] = df['age'].astype('int')",
                        "constraints": ["non-null", "min: 0", "max: 120"], 
                        "example_values": [25, 30]
                    }}
                ]
            }}
    """

    response = model.generate_content(prompt)
    return response.text

def parse_rules(llm_output):
    """Parse LLM output into ValidationRules object with enhanced error handling (replacing Pydantic parsing)."""
    try:
        # Extract JSON if wrapped in markdown code block
        json_match = re.search(r"```json\n(.*?)\n```", llm_output, re.DOTALL)
        parsed_json = json_match.group(1) if json_match else llm_output    
        
        # Parse JSON
        raw_data = json.loads(parsed_json)
        
        # Validate with custom validation
        rules = []
        if "rules" in raw_data:
            for rule_data in raw_data["rules"]:
                # Validate required fields
                if "column" not in rule_data or "data_type" not in rule_data:
                    continue
                    
                # Validate data_type
                if rule_data["data_type"] not in CONFIG['ALLOWED_DATA_TYPES']:
                    rule_data["data_type"] = "string"  # Default to string if invalid
                
                # Create ValidationRule object
                rule = ValidationRule(
                    column=rule_data["column"],
                    data_type=rule_data["data_type"],
                    constraints=rule_data.get("constraints", [])
                )
                rules.append(rule)
        
        return ValidationRules(rules)
    
    except json.JSONDecodeError as e:
        print(f"JSON Decode Error: {str(e)}")
        print(f"LLM Output: {llm_output[:500]}...")  # Print first 500 chars for debugging
        # Return empty validation rules on error
        return ValidationRules([])
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        # Return empty validation rules on error
        return ValidationRules([])

def validate_value(value, rule):
    """Validate a single value based on rule constraints with improved error handling."""
    errors = []
    
    # Skip validation for null values unless non-null constraint exists
    if pd.isnull(value):
        if "non-null" in rule.constraints:
            errors.append("Cannot be null")
        return errors
    
    # Data Type Validation
    expected_type = rule.data_type
    if expected_type == "integer" and not (isinstance(value, int) or (isinstance(value, float) and value.is_integer())):
        errors.append("Expected DataType: integer")
    elif expected_type == "float" and not isinstance(value, (int, float)):
        errors.append("Expected DataType: float")
    elif expected_type == "string" and not isinstance(value, str):
        errors.append("Expected DataType: string")
    elif expected_type == "boolean" and not isinstance(value, bool):
        errors.append("Expected DataType: boolean")
    elif expected_type == "date" and not pd.api.types.is_datetime64_any_dtype(type(value)):
        errors.append("Expected DataType: date")
    
    # Constraint Validation
    for constraint in rule.constraints:
        # Min/Max Check for numeric values
        if isinstance(value, (int, float)):
            if constraint.startswith("min:"):
                min_val = float(constraint.split(":", 1)[1].strip())
                if value < min_val:
                    errors.append(f"Below minimum ({min_val})")
            elif constraint.startswith("max:"):
                max_val = float(constraint.split(":", 1)[1].strip())
                if value > max_val:
                    errors.append(f"Above maximum ({max_val})")
        
        # Regex Check for strings
        if constraint.startswith("regex:") and isinstance(value, str):
            regex_pattern = constraint.split("regex:", 1)[1].strip()
            if not re.match(regex_pattern, value):
                errors.append("Invalid format")
        
        # Unique check is handled separately at dataframe level
    
    return errors

def generate_validation_report(df, validation_rules):
    """Generate a set of DataFrame-style validation reports by test category."""
    # Parse rules
    try:
        if isinstance(validation_rules, str):
            rules = parse_rules(validation_rules)
        else:
            rules = validation_rules
        
        validation_results = defaultdict(lambda: defaultdict(list))
        
        # Collect errors by column
        for rule in rules.rules:
            column_name = rule.column
            if column_name not in df.columns:
                validation_results[column_name]["Missing column"].append("All rows")
                continue
            
            # Check for unique constraint at dataframe level
            if "unique" in rule.constraints and df[column_name].duplicated().any():
                duplicate_indices = df[df[column_name].duplicated(keep=False)].index.tolist()
                validation_results[column_name]["Has duplicates"].append(
                    ", ".join(map(str, duplicate_indices[:10])) + 
                    (f" and {len(duplicate_indices) - 10} more" if len(duplicate_indices) > 10 else "")
                )
            
            # Validate individual values
            for index, value in df[column_name].items():
                errors = validate_value(value, rule)
                for error in errors:
                    validation_results[column_name][error].append(index)
        
        # Create separate report DataFrames by test category
        
        # 1. Data Type Test
        # 1. Data Type Test
        datatype_columns = ['Column Name', 'Expected Type', 'Actual Type', 'Type Match', 'Details']
        datatype_data = []

        def compare_data_types(expected_type, actual_type):
            """Compare expected and actual types"""
            type_match = expected_type == actual_type
            details = ""
            
            if not type_match:
                details = f"Convert to detected type: {expected_type} from the actual detected type: {actual_type}"
            
            return type_match, details

        for rule in rules.rules:
            if rule.column in df.columns:
                # Get actual type
                if hasattr(rule, 'actual_type') and rule.actual_type:
                    actual_type = rule.actual_type
                else:
                    actual_type = str(df[rule.column].dtype)
                    
                # Simple direct comparison
                type_match, details = compare_data_types(rule.data_type, actual_type)
                
                # Add any validation errors from validate_value
                type_errors = [err for err in validation_results[rule.column]
                            if err.startswith("Expected DataType")]
                if type_errors:
                    error_details = "; ".join(f"{err}: indices {', '.join(map(str, validation_results[rule.column][err][:5]))}" 
                                            for err in type_errors)
                    details = error_details if not details else f"{details}; {error_details}"

                datatype_data.append([
                    rule.column,
                    rule.data_type,
                    actual_type,
                    "✅" if type_match else "❌",
                    details
                ])

        # 2. Out of Range Test
        range_columns = ['Column Name', 'Min', 'Max', 'Out of Range', 'Details']
        range_data = []
        
        for rule in rules.rules:
            if rule.column in df.columns:
                # Extract min/max constraints
                min_val = None
                max_val = None
                for constraint in rule.constraints:
                    if constraint.startswith("min:"):
                        min_val = constraint.split(":", 1)[1].strip()
                    elif constraint.startswith("max:"):
                        max_val = constraint.split(":", 1)[1].strip()
                
                # Skip if no range constraints
                if min_val is None and max_val is None:
                    continue
                
                # Check for range errors
                below_min = "Below minimum" in validation_results[rule.column]
                above_max = "Above maximum" in validation_results[rule.column]
                
                # Combine details
                details = ""
                if below_min:
                    indices = validation_results[rule.column]["Below minimum"]
                    details += f"Below min: Row indices {', '.join(map(str, indices[:5]))}"
                    if len(indices) > 5:
                        details += f" and {len(indices) - 5} more"
                    details += "; "
                
                if above_max:
                    indices = validation_results[rule.column]["Above maximum"]
                    details += f"Above max: Row indices {', '.join(map(str, indices[:5]))}"
                    if len(indices) > 5:
                        details += f" and {len(indices) - 5} more"
                
                range_data.append([
                    rule.column,
                    min_val or "N/A",
                    max_val or "N/A",
                    "❌" if (below_min or above_max) else "✅",
                    details
                ])
        
        # 3. Duplicates Test
        duplicate_columns = ['Column Name', 'Unique Constraint', 'Has Duplicates', 'Details']
        duplicate_data = []
        
        for rule in rules.rules:
            if rule.column in df.columns:
                # Check if unique constraint exists
                has_unique_constraint = "unique" in rule.constraints
                
                # Check for duplicates
                has_duplicates = "Has duplicates" in validation_results[rule.column]
                
                # Get details
                details = ""
                if has_duplicates:
                    indices = validation_results[rule.column]["Has duplicates"]
                    # Join any string representation in indices (it's already formatted)
                    details = "".join(str(idx) for idx in indices)
                
                duplicate_data.append([
                    rule.column,
                    "Yes" if has_unique_constraint else "No",
                    "❌" if has_duplicates else "✅",
                    details
                ])
        
        # 4. Missing Values (Null) Test
        null_columns = ['Column Name', 'Non-Null Constraint', 'Has Nulls', 'Details']
        null_data = []
        
        for rule in rules.rules:
            if rule.column in df.columns:
                # Check if non-null constraint exists
                has_nonnull_constraint = "non-null" in rule.constraints
                
                # Check for null values
                has_nulls = "Cannot be null" in validation_results[rule.column]
                
                # Get total nulls
                total_nulls = df[rule.column].isnull().sum()
                null_percent = (total_nulls / len(df)) * 100 if len(df) > 0 else 0
                
                # Details for nulls
                details = ""
                if has_nulls and total_nulls > 0:
                    details = f"{total_nulls} null values ({null_percent:.1f}% of data)"
                    if has_nonnull_constraint:
                        indices = validation_results[rule.column]["Cannot be null"]
                        if indices:
                            details += f" at row indices: {', '.join(map(str, indices[:5]))}"
                            if len(indices) > 5:
                                details += f" and {len(indices) - 5} more"
                
                null_data.append([
                    rule.column,
                    "Yes" if has_nonnull_constraint else "No",
                    "❌" if has_nulls and has_nonnull_constraint else "✅",
                    details
                ])
        
        # 5. Data Formatting Test
        format_columns = ['Column Name', 'Format Constraint', 'Has Format Issues', 'Details']
        format_data = []
        
        for rule in rules.rules:
            if rule.column in df.columns:
                # Check if format constraint exists
                format_constraint = next((c for c in rule.constraints if c.startswith("regex:")), None)
                
                # Check for format issues
                has_format_issues = "Invalid format" in validation_results[rule.column]
                
                # Format constraint display
                constraint_display = "None"
                if format_constraint:
                    # Extract the actual regex pattern, limiting its length for display
                    pattern = format_constraint.split("regex:", 1)[1].strip()
                    if len(pattern) > 30:
                        constraint_display = pattern[:27] + "..."
                    else:
                        constraint_display = pattern
                
                # Details for format issues
                details = ""
                if has_format_issues:
                    indices = validation_results[rule.column]["Invalid format"]
                    details = f"Format issues at row indices: {', '.join(map(str, indices[:5]))}"
                    if len(indices) > 5:
                        details += f" and {len(indices) - 5} more"
                
                # Only add to report if there's a format constraint or format issues
                if format_constraint or has_format_issues:
                    format_data.append([
                        rule.column,
                        constraint_display,
                        "❌" if has_format_issues else "✅",
                        details
                    ])
        
        # Create DataFrames for each test
        data_type_df = pd.DataFrame(datatype_data, columns=datatype_columns)
        range_df = pd.DataFrame(range_data, columns=range_columns)
        duplicate_df = pd.DataFrame(duplicate_data, columns=duplicate_columns)
        null_df = pd.DataFrame(null_data, columns=null_columns)
        format_df = pd.DataFrame(format_data, columns=format_columns)
        
        # Pack all reports into a dictionary
        reports = {
            "data_type": data_type_df,
            "range": range_df,
            "duplicate": duplicate_df,
            "null": null_df,
            "format": format_df
        }
        
        # For backward compatibility, create a combined report
        combined_columns = ['Column Name', 'Type Valid', 'Range Valid', 'No Duplicates', 'Non-Null', 'Format Valid', 'Details']
        combined_data = []
        
        for column in df.columns:
            # Compile results from each test
            type_valid = "✅"
            range_valid = "✅"
            no_duplicates = "✅"
            non_null = "✅"
            format_valid = "✅"
            all_details = []
            
            # Type check
            type_row = data_type_df[data_type_df['Column Name'] == column]
            if not type_row.empty and type_row['Type Match'].values[0] == "❌":
                type_valid = "❌"
                if type_row['Details'].values[0]:
                    all_details.append(type_row['Details'].values[0])
            
            # Range check
            range_row = range_df[range_df['Column Name'] == column]
            if not range_row.empty and range_row['Out of Range'].values[0] == "❌":
                range_valid = "❌"
                if range_row['Details'].values[0]:
                    all_details.append(range_row['Details'].values[0])
            
            # Duplicates check
            dup_row = duplicate_df[duplicate_df['Column Name'] == column]
            if not dup_row.empty and dup_row['Has Duplicates'].values[0] == "❌":
                no_duplicates = "❌"
                if dup_row['Details'].values[0]:
                    all_details.append(dup_row['Details'].values[0])
            
            # Null check
            null_row = null_df[null_df['Column Name'] == column]
            if not null_row.empty and null_row['Has Nulls'].values[0] == "❌":
                non_null = "❌"
                if null_row['Details'].values[0]:
                    all_details.append(null_row['Details'].values[0])
            
            # Format check
            format_row = format_df[format_df['Column Name'] == column]
            if not format_row.empty and format_row['Has Format Issues'].values[0] == "❌":
                format_valid = "❌"
                if format_row['Details'].values[0]:
                    all_details.append(format_row['Details'].values[0])
            
            # Add to combined data
            combined_data.append([
                column,
                type_valid,
                range_valid,
                no_duplicates,
                non_null,
                format_valid,
                "; ".join(all_details)
            ])
        
        combined_df = pd.DataFrame(combined_data, columns=combined_columns)
        
        return reports, combined_df, validation_results
    
    except Exception as e:
        raise ValueError(f"Error generating validation report: {str(e)}")

# ----- FEATURE ANALYSIS -----
def analyze_feature_details(df: pd.DataFrame, feature: str):
    """Generate detailed analysis for a specific feature using custom class (replacing Pydantic model)"""
    analysis = {
        'name': feature,
        'dtype': str(df[feature].dtype),
        'total_count': len(df),
        'null_count': df[feature].isnull().sum(),
        'unique_count': df[feature].nunique(),
        'duplicate_count': len(df) - df[feature].nunique(),
    }
    
    # Add descriptive statistics based on data type
    if np.issubdtype(df[feature].dtype, np.number):
        analysis.update({
            'min_value': df[feature].min(),
            'max_value': df[feature].max(),
            'mean': df[feature].mean(),
            'median': df[feature].median(),
            'std': df[feature].std()
        })
    elif df[feature].dtype == 'object' or pd.api.types.is_categorical_dtype(df[feature]):
        # Get value counts for categorical data
        value_counts = df[feature].value_counts().head(5).to_dict()
        analysis['top_values'] = value_counts
    
    return analysis

#---------------------------------------------------------------------------------------------------------------------------------------------

def generate_ai_summary(schema, validation_reports, combined_report_df, validation_columns):
    """Generate AI summary of validation results"""
    total_checks = calculate_validation_metrics(validation_reports)['total_checks']
    passed_checks = (combined_report_df[validation_columns] == '✅').sum().sum()
    quality_score = (passed_checks / total_checks) * 100 if total_checks > 0 else 0
    
    prompt = f"""
You are a data quality expert. Analyze these validation results:

**Validation Reports**:
- Data Type Issues: {len(validation_reports['data_type'])} columns tested
- Range Issues: {len(validation_reports['range'])} columns tested  
- Duplicates: {len(validation_reports['duplicate'])} columns tested
- Nulls: {len(validation_reports['null'])} columns tested
- Format Issues: {len(validation_reports['format'])} columns tested

**Overall Stats**:
- Quality Score: {quality_score:.1f}%
- {passed_checks} out of {total_checks} checks passed

Please:
1. Summarize what validation tests were performed
2. Identify key problems found
3. Recommend practical solutions

Respond in markdown (no headings), professionally and concisely.
    """
    
    return model.generate_content(prompt).text


#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def calculate_validation_metrics(validation_reports):
    """Calculate total and passed checks for all validation types consistently"""
    metrics = {'total_checks': 0, 'passed_checks': 0, 'type_details': {}}
    
    for report_key, status_column in CONFIG['VALIDATION_MAPPINGS'].items():
        if report_key in validation_reports:
            df = validation_reports[report_key]
            total = len(df)
            passed = len(df[df[status_column] == '✅']) if total > 0 else 0
            
            metrics['total_checks'] += total
            metrics['passed_checks'] += passed
            metrics['type_details'][report_key] = {
                'total': total, 'passed': passed,
                'pass_rate': (passed / total * 100) if total > 0 else 0
            }
    return metrics

def calculate_actual_total_checks(validation_reports, combined_report_df):
    """Calculate actual number of applicable validation checks"""
    return calculate_validation_metrics(validation_reports)['total_checks']

def create_donut_chart(passed, failed, percentage):
    """Create a standardized donut chart for validation metrics"""
    fig = go.Figure(data=[go.Pie(
        labels=['Pass', 'Fail'],
        values=[passed, failed],
        hole=.6,
        marker_colors=['#28a745', '#dc3545']
    )])
    
    fig.update_layout(
        annotations=[dict(text=f'{percentage:.1f}%', x=0.5, y=0.5, font_size=20, showarrow=False)],
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        margin=dict(t=0, l=0, r=0, b=0),
        height=200
    )
    return fig

def apply_custom_css():
    """Apply custom CSS styling to the Streamlit app"""
    st.markdown("""
    <style>
    .main .block-container {padding-top: 2rem;}
    h1, h2, h3 {color: #1E88E5;}
    .stTabs [data-baseweb="tab-list"] {gap: 2px; background-color: #f0f2f6; border-radius: 4px;}
    .stTabs [data-baseweb="tab"] {padding: 10px 20px; border-radius: 4px 4px 0 0; background-color: #f0f2f6;}
    .stTabs [aria-selected="true"] {background-color: #1E88E5 !important; color: white !important;}
    .stDataFrame {border-radius: 8px; overflow: hidden;}
    [data-testid="stMetricValue"] {font-size: 1.8rem !important; color: #1E88E5 !important; font-weight: bold !important;}
    </style>
    """, unsafe_allow_html=True)


def main():
    """Main function for the Data Quality Assessment App"""
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Configure the Streamlit page
    st.set_page_config(
        page_title="Data Quality Assessment App", 
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Apply custom styling
    apply_custom_css()
    
    # Create sidebar
    with st.sidebar:
        st.title("📊 Data Quality Assessment Tool")
        st.markdown("---")
        
        uploaded_file = st.file_uploader("Upload Dataset", type=['csv', 'json', 'xlsx', 'xls'])
        
        # Add app info
        st.sidebar.markdown("---")
        st.sidebar.markdown("### User Info")
        st.sidebar.markdown(f"**Time:** {current_time}")
        st.sidebar.markdown(f"**User:** {CONFIG['CURRENT_USER']}")
    
    # Main container
    st.title("📈 Data Quality Assessment Application")
    st.markdown("Upload a dataset to begin analysis and validation.")
    
    if uploaded_file is not None:
        try:
                # Load and analyze data
                with st.spinner("Loading data..."):
                    df = load_data(uploaded_file)
                    
                    # Create a success message with file info
                    #st.success(f"Successfully loaded: **{uploaded_file.name}** ({len(df)} rows × {len(df.columns)} columns)")
                
                # Create tabs for different sections
                #tabs = st.tabs(["📋 Overview", "🔍 Data Validation"])
                st.subheader("Data Preview")
                with st.expander('Data Preview'):
                        #st.header("Dataset Overview")
                        
                    # Summary metrics in modern cards
                    col1, col2 = st.columns(2)
                    metrics_data = [
                        {"title": "Total Rows", "value": len(df), "icon": "📊"},
                        {"title": "Total Columns", "value": len(df.columns), "icon": "📋"},
                        #{"title": "Missing Values", "value": df.isnull().sum().sum(), "icon": "❓"},
                        #{"title": "Duplicate Rows", "value": df.duplicated().sum(), "icon": "🔄"}
                    ]
                    
                    for col, metric in zip([col1, col2], metrics_data):
                        with col:
                            st.markdown(f"""
                            <div style="border: 1px solid #ddd; 
                                        border-radius: 8px; 
                                        padding: 15px; 
                                        text-align: center;
                                        background: white;
                                        box-shadow: 0 2px 4px rgba(0,0,0,0.05);">
                                <div style="font-size: 24px; margin-bottom: 5px;">{metric['icon']}</div>
                                <div style="color: #666; font-size: 14px;">{metric['title']}</div>
                                <div style="font-size: 24px; font-weight: bold; color: #2a5298; margin-top: 5px;">
                                    {metric['value']:,}
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    # Data Preview in a modern card
                    st.markdown("""
                    <div style="margin: 30px 0 10px 0;">
                        <div style="font-size: 14px; color: #666;">First 10 rows of the dataset</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                    st.dataframe(df.head(10), use_container_width=True)
                    st.markdown("</div>", unsafe_allow_html=True)
                
                st.header("Data Validation Assessment")
                
                # Generate schema and validation rules
                with st.spinner("Analyzing data schema and generating validation rules..."):
                    schema = profile_data(df)
                    suggestions = generate_schema_suggestions(schema)
                    validation_rules = parse_rules(suggestions)
                    validation_reports, combined_report_df, validation_results = generate_validation_report(df, validation_rules)
                
                # Create subtabs for different validation categories
                validation_tabs = st.tabs([
                    "📑 Assessment Summary",
                    "📊 Data Type Test", 
                    "📏 Range Test", 
                    "🔄 Duplicates Test", 
                    "❓ Missing Values Test", 
                    "🔤 Format Test",
                    
                ])
                

                with validation_tabs[0]:
                    #st.subheader("Validation Summary")
                    
                    if not combined_report_df.empty:
                        # Calculate overall quality score using unified function
                        validation_columns = ['Type Valid', 'Range Valid', 'No Duplicates', 'Non-Null', 'Format Valid']
                        validation_metrics = calculate_validation_metrics(validation_reports)
                        total_checks = validation_metrics['total_checks']
                        passed_checks = validation_metrics['passed_checks']
                        
                        quality_score = (passed_checks / total_checks) * 100 if total_checks > 0 else 0
                        columns_with_issues = (combined_report_df[validation_columns] == '❌').any(axis=1).sum()
                        
                        # Create two columns with 2:1 ratio
                        col_summary, col_charts = st.columns([2, 1])
                        
                        with col_summary:
                            # Add loading indicator for summary
                            with st.spinner('Generating validation summary...'):
                                # Add scrollable container for executive summary with increased height to match pie chart
                                schema = profile_data(df)
                                suggestions = generate_schema_suggestions(schema)
                                validation_rules = parse_rules(suggestions)
                                validation_reports, combined_report_df, validation_results = generate_validation_report(df, validation_rules)
                                response = generate_ai_summary(schema, validation_reports, combined_report_df, validation_columns)
                                
                                st.markdown(f"""
                                <div style="height: 430px; overflow-y: auto; padding-right: 10px; margin-bottom: 20px;">
                                    <div style="background: white; 
                                                padding: 20px; 
                                                border-radius: 8px; 
                                                border: 1px solid #ddd;
                                                box-shadow: 0 2px 4px rgba(0,0,0,0.05);
                                                min-height: 280px;">  <!-- Set min-height to match pie chart height -->
                                        <div style="color: #1a1a1a; line-height: 1.6;">
                                            {response}
                                        </div>
                                    </div>
                                </div>
                                """, unsafe_allow_html=True)

                        with col_charts:
                            # Create donut chart
                            fig = create_donut_chart(passed_checks, total_checks - passed_checks, quality_score)
                            st.plotly_chart(fig, use_container_width=True, key="summary_chart")

                            # Add spacing after pie chart
                            st.markdown("<div style='margin-bottom: 30px;'></div>", unsafe_allow_html=True)

                            # Status boxes
                            st.markdown(f"""
                            <div style="display: flex; border: 1px solid lightgray; border-radius: 8px; overflow: hidden;">
                                <div style="background-color: #d4edda; color: #155724; padding: 15px; flex: 1; text-align: center;">
                                    <div style="font-size: 20px; font-weight: bold;">✅ Passed</div>
                                    <div style="font-size: 32px; font-weight: bold;">{passed_checks}</div>
                                    <div style="font-size: 12px;">({quality_score:.1f}%)</div>
                                </div>
                                <div style="background-color: #f8d7da; color: #721c24; padding: 15px; flex: 1; text-align: center;">
                                    <div style="font-size: 20px; font-weight: bold;">❌ Failed</div>
                                    <div style="font-size: 32px; font-weight: bold;">{total_checks - passed_checks}</div>
                                    <div style="font-size: 12px;\">({100-quality_score:.1f}%)</div>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)

                        # Add spacing before validation type summaries
                        st.markdown("<div style='margin-top: 40px;'></div>", unsafe_allow_html=True)

                        # Calculate and display validation type summaries using unified metrics
                        validation_summaries = []
                        validation_metrics = calculate_validation_metrics(validation_reports)
                        
                        # Map validation columns to their corresponding report keys
                        validation_type_mapping = {
                            'Type Valid': 'data_type',
                            'Range Valid': 'range', 
                            'No Duplicates': 'duplicate',
                            'Non-Null': 'null',
                            'Format Valid': 'format'
                        }
                        
                        for validation_type in validation_columns:
                            report_key = validation_type_mapping.get(validation_type)
                            if report_key and report_key in validation_metrics['type_details']:
                                details = validation_metrics['type_details'][report_key]
                                validation_summaries.append({
                                    'type': validation_type,
                                    'passed': details['passed'],
                                    'total': details['total'],
                                    'rate': details['pass_rate']
                                })
                            else:
                                # Fallback if report not found
                                validation_summaries.append({
                                    'type': validation_type,
                                    'passed': 0,
                                    'total': 0,
                                    'rate': 0.0
                                })
                        
                        # Display validation type summaries in a grid
                        for idx in range(0, len(validation_summaries), 3):
                            cols = st.columns(3)
                            for col_idx, col in enumerate(cols):
                                if idx + col_idx < len(validation_summaries):
                                    summary = validation_summaries[idx + col_idx]
                                    col.markdown(f"""
                                    <div style="border: 1px solid #ddd; border-radius: 8px; padding: 15px; margin: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.05);">
                                        <div style="font-weight: bold; margin-bottom: 10px; color: #333; font-size: 18px;">{summary['type']}</div>
                                        <div style="display: flex; justify-content: space-between; margin-bottom: 10px;">
                                            <div style="text-align: center; flex: 1;">
                                                <div style="color: #666; font-size: 12px;">Pass Rate</div>
                                                <div style="font-size: 28px; font-weight: bold; color: {'#28a745' if summary['rate'] >= 80 else '#dc3545'};">
                                                    {summary['rate']:.1f}%
                                                </div>
                                            </div>
                                        </div>
                                        <div style="background-color: #f8f9fa; padding: 8px; border-radius: 6px; text-align: center;">
                                            <div style="color: #666; font-size: 16px;">Passed/Total</div>
                                            <div style="font-weight: bold; color: #2a5298;">
                                                {summary['passed']}/{summary['total']}
                                            </div>
                                        </div>
                                    </div>
                                    """, unsafe_allow_html=True)
                    else:
                        st.info("No validation results available.")

                # 1. Data Type Test Tab
                with validation_tabs[1]:
                    st.subheader("Data Type Validation")
                    #st.write("This test verifies if the data types match the expected types.")
                    
                    type_df = validation_reports["data_type"]
                    if not type_df.empty:
                        # Calculate metrics
                        total_columns = len(type_df)
                        matching_types = len(type_df[type_df['Type Match'] == '✅'])
                        mismatched_types = len(type_df[type_df['Type Match'] == '❌'])
                        match_percentage = (matching_types / total_columns) * 100
                        
                        # Progress visualization with Pie Chart
                        col1, col2 = st.columns([3, 2])
                        with col1:
                            # Create donut chart
                            fig = create_donut_chart(matching_types, mismatched_types, match_percentage)
                            fig.update_layout(height=200)
                            st.plotly_chart(fig, use_container_width=True, key="datatype_chart")

                        with col2:
                            # Status boxes
                            st.markdown(f"""
                            <div style="display: flex; border: 1px solid lightgray; border-radius: 8px; overflow: hidden; margin-top: 20px;">
                                <div style="background-color: #d4edda; color: #155724; padding: 20px; flex: 1; text-align: center;">
                                    <div style="font-size: 24px; font-weight: bold;">✅ Passed</div>
                                    <div style="font-size: 36px; font-weight: bold;">{matching_types}</div>
                                    <div style="font-size: 14px;">({match_percentage:.1f}%)</div>
                                </div>
                                <div style="background-color: #f8d7da; color: #721c24; padding: 20px; flex: 1; text-align: center;">
                                    <div style="font-size: 24px; font-weight: bold;">❌ Failed</div>
                                    <div style="font-size: 36px; font-weight: bold;">{mismatched_types}</div>
                                    <div style="font-size: 14px;">({100-match_percentage:.1f}%)</div>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)

                        # Type Breakdown with split metrics in expander
                        with st.expander("Column-wise Type Analysis", expanded=True):
                            # Create a grid of metrics for each column (3 columns)
                            for idx in range(0, len(type_df), 3):
                                cols = st.columns(3)
                                
                                # Process three rows at a time
                                for col_idx in range(3):
                                    if idx + col_idx < len(type_df):
                                        row = type_df.iloc[idx + col_idx]
                                        is_match = row['Type Match'] == '✅'
                                        with cols[col_idx]:
                                            st.markdown(f"""
                                            <div style="border: 1px solid #ddd; border-radius: 8px; padding: 15px; margin: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.05);">
                                                <div style="font-weight: bold; margin-bottom: 10px; color: #333; font-size: 16px;">{row['Column Name']}</div>
                                                <div style="display: flex; justify-content: space-between; gap: 10px;">
                                                    <div style="text-align: center; flex: 1; background-color: #f8f9fa; padding: 10px; border-radius: 6px; border: 1px solid #dee2e6;">
                                                        <div style="color: #495057; font-size: 14px;">Actual</div>
                                                        <div style="font-weight: bold; color: #495057; font-size: 16px;">{row['Actual Type']}</div>
                                                    </div>
                                                    <div style="text-align: center; flex: 1; background-color: {'#d4edda' if is_match else '#f8d7da'}; padding: 10px; border-radius: 6px; border: 1px solid {'#c3e6cb' if is_match else '#f5c6cb'};">
                                                        <div style="color: {'#155724' if is_match else '#721c24'}; font-size: 14px;">Expected</div>
                                                        <div style="font-weight: bold; color: {'#155724' if is_match else '#721c24'}; font-size: 16px;">{row['Expected Type']}</div>
                                                    </div>
                                                </div>
                                                <div style="text-align: center; margin-top: 12px;">
                                                    <span style="padding: 4px 12px; border-radius: 12px; font-size: 14px; 
                                                            background-color: {'#d4edda' if is_match else '#f8d7da'}; 
                                                            color: {'#155724' if is_match else '#721c24'};
                                                            border: 1px solid {'#c3e6cb' if is_match else '#f5c6cb'};
                                                            font-weight: 500;">
                                                        {row['Type Match']}
                                                    </span>
                                                </div>
                                            </div>
                                            """, unsafe_allow_html=True)

                    else:
                        st.info("No data type validation rules available.")
                   
                # 2. Range Test Tab
                with validation_tabs[2]:
                    st.subheader("Numeric Range Validation")
                    #st.write("This test checks if numeric values are within the specified ranges.")
                    
                    range_df = validation_reports["range"]
                    if not range_df.empty:
                        # Calculate metrics
                        total_columns = len(range_df)
                        in_range = len(range_df[range_df['Out of Range'] == '✅'])
                        out_range = len(range_df[range_df['Out of Range'] == '❌'])
                        in_range_percentage = (in_range / total_columns) * 100
                        
                        # Progress visualization with Pie Chart
                        col1, col2 = st.columns([3, 2])
                        with col1:
                            # Create donut chart
                            fig = create_donut_chart(in_range, out_range, in_range_percentage)
                            fig.update_layout(height=200)
                            st.plotly_chart(fig, use_container_width=True, key="range_chart")

                        with col2:
                            # Status boxes
                            st.markdown(f"""
                            <div style="display: flex; border: 1px solid lightgray; border-radius: 8px; overflow: hidden; margin-top: 20px;">
                                <div style="background-color: #d4edda; color: #155724; padding: 20px; flex: 1; text-align: center;">
                                    <div style="font-size: 24px; font-weight: bold;">✅ Within Range</div>
                                    <div style="font-size: 36px; font-weight: bold;">{in_range}</div>
                                    <div style="font-size: 14px;">({in_range_percentage:.1f}%)</div>
                                </div>
                                <div style="background-color: #f8d7da; color: #721c24; padding: 20px; flex: 1; text-align: center;">
                                    <div style="font-size: 24px; font-weight: bold;">❌ Out of Range</div>
                                    <div style="font-size: 36px; font-weight: bold;">{out_range}</div>
                                    <div style="font-size: 14px;">({100-in_range_percentage:.1f}%)</div>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)

                        # Range Analysis with split metrics in expander
                        with st.expander("Column-wise Range Analysis", expanded=True):
                            # Create a grid of metrics for each column (3 columns)
                            for idx in range(0, len(range_df), 3):
                                cols = st.columns(3)
                                
                                # Process three rows at a time
                                for col_idx in range(3):
                                    if idx + col_idx < len(range_df):
                                        row = range_df.iloc[idx + col_idx]
                                        is_in_range = row['Out of Range'] == '✅'
                                        with cols[col_idx]:
                                            st.markdown(f"""
                                            <div style="border: 1px solid #ddd; border-radius: 8px; padding: 15px; margin: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.05);">
                                                <div style="font-weight: bold; margin-bottom: 10px; color: #333; font-size: 16px;">{row['Column Name']}</div>
                                                <div style="display: flex; flex-direction: column; gap: 10px;">
                                                    <div style="display: flex; gap: 10px;">
                                                        <div style="text-align: center; flex: 1; background-color: #f8f9fa; padding: 10px; border-radius: 6px;">
                                                            <div style="color: #666; font-size: 14px;">Min</div>
                                                            <div style="font-weight: bold; font-size: 16px;">{row['Min']}</div>
                                                        </div>
                                                        <div style="text-align: center; flex: 1; background-color: #f8f9fa; padding: 10px; border-radius: 6px;">
                                                            <div style="color: #666; font-size: 14px;">Max</div>
                                                            <div style="font-weight: bold; font-size: 16px;">{row['Max']}</div>
                                                        </div>
                                                    </div>
                                                    <div style="background-color: {'#d4edda' if is_in_range else '#f8d7da'}; 
                                                                padding: 10px; 
                                                                border-radius: 6px;
                                                                border: 1px solid {'#c3e6cb' if is_in_range else '#f5c6cb'};">
                                                        <div style="color: {'#155724' if is_in_range else '#721c24'}; 
                                                                    font-size: 14px; 
                                                                    margin-bottom: 5px;
                                                                    display: flex;
                                                                    justify-content: space-between;
                                                                    align-items: center;">
                                                            <span>Status</span>
                                                            <span>{row['Out of Range']}</span>
                                                        </div>
                                                        <div style="color: {'#155724' if is_in_range else '#721c24'}; 
                                                                    font-size: 14px; 
                                                                    font-family: monospace;
                                                                    display: block;
                                                                    text-align: left;">
                                                            {row['Details'] if row['Details'] else 'All values within range'}
                                                        </div>
                                                    </div>
                                                </div>
                                            </div>
                                            """, unsafe_allow_html=True)
                    else:
                        st.info("No range validation rules available.")
                
                # 3. Duplicates Test Tab
                with validation_tabs[3]:
                    st.subheader("Duplicates Validation")
                    #st.write("This test checks for duplicate values in columns with uniqueness constraints.")
                    
                    duplicate_df = validation_reports["duplicate"]
                    if not duplicate_df.empty:
                        # Calculate metrics
                        total_columns = len(duplicate_df)
                        unique_cols = len(duplicate_df[duplicate_df['Has Duplicates'] == '✅'])
                        duplicate_cols = len(duplicate_df[duplicate_df['Has Duplicates'] == '❌'])
                        unique_percentage = (unique_cols / total_columns) * 100
                        
                        # Progress visualization with Pie Chart
                        col1, col2 = st.columns([3, 2])
                        with col1:
                            # Create donut chart
                            fig = create_donut_chart(unique_cols, duplicate_cols, unique_percentage)
                            fig.update_layout(height=200)
                            st.plotly_chart(fig, use_container_width=True, key="duplicates_chart")

                        with col2:
                            # Status boxes
                            st.markdown(f"""
                            <div style="display: flex; border: 1px solid lightgray; border-radius: 8px; overflow: hidden; margin-top: 20px;">
                                <div style="background-color: #d4edda; color: #155724; padding: 20px; flex: 1; text-align: center;">
                                    <div style="font-size: 24px; font-weight: bold;">✅ Unique</div>
                                    <div style="font-size: 36px; font-weight: bold;">{unique_cols}</div>
                                    <div style="font-size: 14px;">({unique_percentage:.1f}%)</div>
                                </div>
                                <div style="background-color: #f8d7da; color: #721c24; padding: 20px; flex: 1; text-align: center;">
                                    <div style="font-size: 24px; font-weight: bold;">❌ Duplicates</div>
                                    <div style="font-size: 36px; font-weight: bold;">{duplicate_cols}</div>
                                    <div style="font-size: 14px;">({100-unique_percentage:.1f}%)</div>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)


                        # Duplicates Analysis with split metrics in expander
                        with st.expander("Column-wise Duplicates Analysis", expanded=True):
                            
                            # Create a grid of metrics for each column (3 columns)
                            for idx in range(0, len(duplicate_df), 3):
                                cols = st.columns(3)
                                
                                # Process three rows at a time
                                for col_idx in range(3):
                                    if idx + col_idx < len(duplicate_df):
                                        row = duplicate_df.iloc[idx + col_idx]
                                        is_unique = row['Has Duplicates'] == '✅'
                                        with cols[col_idx]:
                                            st.markdown(f"""
                                            <div style="border: 1px solid #ddd; border-radius: 8px; padding: 15px; margin: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.05);">
                                                <div style="font-weight: bold; margin-bottom: 10px; color: #333; font-size: 16px;">{row['Column Name']}</div>
                                                <div style="display: flex; flex-direction: column; gap: 10px;">
                                                    <div style="text-align: center; background-color: #f8f9fa; padding: 10px; border-radius: 6px;">
                                                        <div style="color: #666; font-size: 14px;">Unique Constraint</div>
                                                        <div style="font-weight: bold; color: #2a5298; font-size: 16px;">{row['Unique Constraint']}</div>
                                                    </div>
                                                    <div style="background-color: {'#d4edda' if is_unique else '#f8d7da'}; 
                                                                padding: 10px; 
                                                                border-radius: 6px;
                                                                border: 1px solid {'#c3e6cb' if is_unique else '#f5c6cb'};">
                                                        <div style="color: {'#155724' if is_unique else '#721c24'}; 
                                                                    font-size: 14px; 
                                                                    margin-bottom: 5px;
                                                                    display: flex;
                                                                    justify-content: space-between;
                                                                    align-items: center;">
                                                            <span>Status</span>
                                                            <span>{row['Has Duplicates']}</span>
                                                        </div>
                                                        <div style="color: {'#155724' if is_unique else '#721c24'}; 
                                                                    font-size: 14px; 
                                                                    font-family: monospace;
                                                                    display: block;
                                                                    text-align: left;">
                                                            {row['Details'] if row['Details'] else 'No duplicates found'}
                                                        </div>
                                                    </div>
                                                </div>
                                            </div>
                                            """, unsafe_allow_html=True)
                    else:
                        st.info("No duplicate validation rules available.")
                
                # 4. Missing Values (Null) Test Tab
                with validation_tabs[4]:
                    st.subheader("Missing Values Validation")
                    #st.write("This test checks for null values in columns with non-null constraints.")
                    
                    null_df = validation_reports["null"]
                    if not null_df.empty:
                        # Calculate metrics
                        total_columns = len(null_df)
                        no_nulls = len(null_df[null_df['Has Nulls'] == '✅'])
                        has_nulls = len(null_df[null_df['Has Nulls'] == '❌'])
                        no_nulls_percentage = (no_nulls / total_columns) * 100
                        
                        # Progress visualization with Pie Chart
                        col1, col2 = st.columns([3, 2])
                        with col1:
                            # Create donut chart
                            fig = create_donut_chart(no_nulls, has_nulls, no_nulls_percentage)
                            fig.update_layout(height=200)
                            st.plotly_chart(fig, use_container_width=True, key="nulls_chart")

                        with col2:
                            # Status boxes
                            st.markdown(f"""
                            <div style="display: flex; border: 1px solid lightgray; border-radius: 8px; overflow: hidden; margin-top: 20px;">
                                <div style="background-color: #d4edda; color: #155724; padding: 20px; flex: 1; text-align: center;">
                                    <div style="font-size: 24px; font-weight: bold;">✅ No Nulls</div>
                                    <div style="font-size: 36px; font-weight: bold;">{no_nulls}</div>
                                    <div style="font-size: 14px;">({no_nulls_percentage:.1f}%)</div>
                                </div>
                                <div style="background-color: #f8d7da; color: #721c24; padding: 20px; flex: 1; text-align: center;">
                                    <div style="font-size: 24px; font-weight: bold;">❌ Has Nulls</div>
                                    <div style="font-size: 36px; font-weight: bold;">{has_nulls}</div>
                                    <div style="font-size: 14px;">({100-no_nulls_percentage:.1f}%)</div>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)


                        # Column Analysis with split metrics in expander
                        with st.expander("Column-wise Null Analysis", expanded=True):
                            # Create a grid of metrics for each column (3 columns)
                            for idx in range(0, len(null_df), 3):
                                cols = st.columns(3)
                                
                                # Process three rows at a time
                                for col_idx in range(3):
                                    if idx + col_idx < len(null_df):
                                        row = null_df.iloc[idx + col_idx]
                                        is_valid = row['Has Nulls'] == '✅'
                                        
                                        # Calculate null values for this column
                                        column_name = row['Column Name']
                                        null_count = df[column_name].isnull().sum()
                                        null_percent = (null_count / len(df)) * 100
                                        
                                        # Determine color based on percentage
                                        if null_percent > 20:
                                            color = "#dc3545"  # Red
                                        elif null_percent > 5:
                                            color = "#ffc107"  # Yellow
                                        else:
                                            color = "#28a745"  # Green
                                        
                                        with cols[col_idx]:
                                            st.markdown(f"""
                                            <div style="border: 1px solid #ddd; border-radius: 8px; padding: 15px; margin: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.05);">
                                                <div style="font-weight: bold; margin-bottom: 10px; color: #333; font-size: 16px;">{column_name}</div>
                                                <div style="display: flex; flex-direction: column; gap: 10px;">
                                                    <div style="display: flex; gap: 10px;">
                                                        <div style="text-align: center; flex: 1; background-color: white; padding: 10px; border-radius: 6px; border: 1px solid #ddd;">
                                                            <div style="color: #666; font-size: 14px;">Null Count</div>
                                                            <div style="font-size: 24px; font-weight: bold; color: {color};">{null_count}</div>
                                                        </div>
                                                        <div style="text-align: center; flex: 1; background-color: white; padding: 10px; border-radius: 6px; border: 1px solid #ddd;">
                                                            <div style="color: #666; font-size: 14px;">Null %</div>
                                                            <div style="font-size: 24px; font-weight: bold; color: {color};">{null_percent:.1f}%</div>
                                                        </div>
                                                    </div>
                                                    <div style="background-color: {'#d4edda' if is_valid else '#f8d7da'}; 
                                                                padding: 10px; 
                                                                border-radius: 6px;
                                                                border: 1px solid {'#c3e6cb' if is_valid else '#f5c6cb'};">
                                                        <div style="color: {'#155724' if is_valid else '#721c24'}; 
                                                                    font-size: 14px; 
                                                                    margin-bottom: 5px;
                                                                    display: flex;
                                                                    justify-content: space-between;
                                                                    align-items: center;">
                                                            <span>Validation Status</span>
                                                            <span>{row['Has Nulls']}</span>
                                                        </div>
                                                        <div style="color: {'#155724' if is_valid else '#721c24'}; 
                                                                    font-size: 14px; 
                                                                    font-family: monospace;
                                                                    text-align: left;
                                                                    display: block;">
                                                            {row['Details'] if row['Details'] else 'No null values found'}
                                                        </div>
                                                    </div>
                                                </div>
                                            </div>
                                            """, unsafe_allow_html=True)

                    else:
                        st.info("No missing value validation rules available.")
                
                # 5. Format Test Tab
                with validation_tabs[5]:
                    st.subheader("Data Format Validation")
                    #st.write("This test checks if values match specified format patterns (e.g., regex).")
                    
                    format_df = validation_reports["format"]
                    if not format_df.empty:
                        # Calculate metrics
                        total_columns = len(format_df)
                        valid_format = len(format_df[format_df['Has Format Issues'] == '✅'])
                        invalid_format = len(format_df[format_df['Has Format Issues'] == '❌'])
                        valid_percentage = (valid_format / total_columns) * 100
                        
                        # Progress visualization with Pie Chart
                        col1, col2 = st.columns([3, 2])
                        with col1:
                            # Create donut chart
                            fig = create_donut_chart(valid_format, invalid_format, valid_percentage)
                            fig.update_layout(height=200)
                            st.plotly_chart(fig, use_container_width=True, key="format_chart")

                        with col2:
                            # Status boxes
                            st.markdown(f"""
                            <div style="display: flex; border: 1px solid lightgray; border-radius: 8px; overflow: hidden; margin-top: 20px;">
                                <div style="background-color: #d4edda; color: #155724; padding: 20px; flex: 1; text-align: center;">
                                    <div style="font-size: 24px; font-weight: bold;">✅ Valid</div>
                                    <div style="font-size: 36px; font-weight: bold;">{valid_format}</div>
                                    <div style="font-size: 14px;">({valid_percentage:.1f}%)</div>
                                </div>
                                <div style="background-color: #f8d7da; color: #721c24; padding: 20px; flex: 1; text-align: center;">
                                    <div style="font-size: 24px; font-weight: bold;">❌ Invalid</div>
                                    <div style="font-size: 36px; font-weight: bold;">{invalid_format}</div>
                                    <div style="font-size: 14px;">({100-valid_percentage:.1f}%)</div>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)

                        # Format Analysis with split metrics in expander
                        with st.expander("Column-wise Format Analysis", expanded=True):
                            # Create a grid of metrics for each column (3 columns)
                            for idx in range(0, len(format_df), 3):
                                cols = st.columns(3)
                                
                                # Process three rows at a time
                                for col_idx in range(3):
                                    if idx + col_idx < len(format_df):
                                        row = format_df.iloc[idx + col_idx]
                                        is_valid = row['Has Format Issues'] == '✅'
                                        with cols[col_idx]:
                                            st.markdown(f"""
                                            <div style="border: 1px solid #ddd; border-radius: 8px; padding: 15px; margin: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.05);">
                                                <div style="font-weight: bold; margin-bottom: 10px; color: #333; font-size: 16px;">{row['Column Name']}</div>
                                                <div style="display: flex; flex-direction: column; gap: 10px;">
                                                    <div style="background-color: #f8f9fa; padding: 12px; border-radius: 6px; border: 1px solid #e9ecef;">
                                                        <div style="color: #666; font-size: 14px; margin-bottom: 4px;">Format Pattern</div>
                                                        <div style="font-family: monospace; 
                                                                    font-size: 15px; 
                                                                    color: #2a5298;
                                                                    word-break: break-all;
                                                                    line-height: 1.4;">
                                                            {row['Format Constraint']}
                                                        </div>
                                                    </div>
                                                    <div style="background-color: {'#d4edda' if is_valid else '#f8d7da'}; 
                                                                padding: 12px; 
                                                                border-radius: 6px;
                                                                border: 1px solid {'#c3e6cb' if is_valid else '#f5c6cb'};">
                                                        <div style="color: {'#155724' if is_valid else '#721c24'}; 
                                                                    font-size: 14px; 
                                                                    margin-bottom: 5px;
                                                                    display: flex;
                                                                    justify-content: space-between;
                                                                    align-items: center;">
                                                            <span>Format Validation</span>
                                                            <span>{row['Has Format Issues']}</span>
                                                        </div>
                                                        <div style="color: {'#155724' if is_valid else '#721c24'}; 
                                                                    font-size: 14px; 
                                                                    font-family: monospace;
                                                                    display: block;
                                                                    text-align: left;">
                                                            {row['Details'] if row['Details'] else 'All values match the format pattern'}
                                                        </div>
                                                    </div>
                                                </div>
                                            </div>
                                            """, unsafe_allow_html=True)

                    else:
                        st.info("No format validation rules available.")
        
        except Exception as e:
            st.error(f"Error processing dataset: {str(e)}")
            st.info("Please upload a valid dataset in CSV, JSON, or Excel format.")

if __name__ == "__main__":
    main()