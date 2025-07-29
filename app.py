import streamlit as st
import os
import pandas as pd
import matplotlib.pyplot as plt
import io
import google.generativeai as genai
from dotenv import load_dotenv
import json
from IPython.display import display, Markdown, JSON
import re
from collections import defaultdict
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_LEFT
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import base64
from matplotlib.backends.backend_agg import FigureCanvasAgg
from typing import List, Dict, Any, Optional, Union
import numpy as np
import tempfile
import uuid
import markdown2
import plotly.graph_objects as go
import graphviz



# Load environment variables
load_dotenv(".env")

# Initialize Gemini client
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel('gemini-2.5-pro')


# Current timestamp and user
CURRENT_TIME = datetime.now()

CURRENT_USER = "Ayaz"


# ----- DATA MODELS (REPLACING PYDANTIC) -----

# Constants for validation
ALLOWED_CONSTRAINT_TYPES = ['min', 'max', 'regex', 'non-null', 'unique', 'enum']
ALLOWED_DATA_TYPES = ['string', 'integer', 'float', 'date', 'boolean', 'category']

class ValidationRule:
    """Class for validation rules (replacing Pydantic ValidationRule)"""
    def __init__(self, column, data_type, actual_type=None, type_match=None, 
                 type_conversion_needed=None, conversion_recommendation=None, 
                 constraints=None, example_values=None):
        self.column = column
        
        # Validate data_type
        if data_type not in ALLOWED_DATA_TYPES:
            raise ValueError(f"Data type must be one of {ALLOWED_DATA_TYPES}")
        self.data_type = data_type
        
        self.actual_type = actual_type
        self.type_match = type_match
        self.type_conversion_needed = type_conversion_needed
        self.conversion_recommendation = conversion_recommendation
        self.constraints = constraints or []
        self.example_values = example_values or []

class ValidationRules:
    """Class for all validation rules (replacing Pydantic ValidationRules)"""
    def __init__(self, rules):
        self.rules = rules

class FeatureAnalysis:
    """Class for feature analysis results (replacing Pydantic FeatureAnalysis)"""
    def __init__(self, name, dtype, total_count, null_count, unique_count, duplicate_count,
                min_value=None, max_value=None, mean=None, median=None, std=None, top_values=None):
        self.name = name
        self.dtype = dtype
        self.total_count = total_count
        self.null_count = null_count
        self.unique_count = unique_count
        self.duplicate_count = duplicate_count
        self.min_value = min_value
        self.max_value = max_value
        self.mean = mean
        self.median = median
        self.std = std
        self.top_values = top_values

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
                if rule_data["data_type"] not in ALLOWED_DATA_TYPES:
                    rule_data["data_type"] = "string"  # Default to string if invalid
                
                # Create ValidationRule object
                rule = ValidationRule(
                    column=rule_data["column"],
                    data_type=rule_data["data_type"],
                    actual_type=rule_data.get("actual_type"),
                    type_match=rule_data.get("type_match"),
                    type_conversion_needed=rule_data.get("type_conversion_needed"),
                    conversion_recommendation=rule_data.get("conversion_recommendation"),
                    constraints=rule_data.get("constraints", []),
                    example_values=rule_data.get("example_values", [])
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
    
    return FeatureAnalysis(**analysis)

#---------------------------------------------------------------------------------------------------------------------------------------------

def generate_datatype_summaries(schema):
    """Generate AI summaries for data type analysis section"""
    suggestion = generate_schema_suggestions(schema)
    
    prompt = f"""
        As a data quality expert, analyze the data schema issues for this dataset by comparing the original schema: {schema} with the expected schema: {suggestion}. Highlight critical type mismatches that may affect analysis, provide summary for the results and discuss the potential data quality impact of these mismatches. Format your response in a professional, concise, and mature way, keeping it under 300 words, and generate the response in markdown and don't include markdown heading in the response.
    """
    
    response = model.generate_content(prompt)
    return response.text


def generate_metrics_summaries(filtered_df, total_checks, passed_checks, quality_score):
    """Generate AI summaries for validation metrics section"""
    prompt = f"""
        As a data quality expert, analyze these data quality metrics for a dataset with a quality score of {quality_score:.1f}%, {passed_checks} out of {total_checks} validation checks passed, and {filtered_df.iloc[:, 1:-1].eq('❌').any(axis=1).sum()} columns with issues. Provide an overall data quality assessment, and summary of the results and discuss the potential business impact of these quality issues. Format your response in a professional, concise, and mature way, keeping it under 300 words, and generate the response in markdown and don't include markdown heading in the response.
    """
    
    response = model.generate_content(prompt)
    return response.text


def generate_nullvalue_summaries(null_value_data):
    """Generate AI summaries for null value analysis section"""
    prompt = f"""
        As a data quality expert, analyze the null value distribution for this dataset: {null_value_data}. Identify columns with concerning null percentages, provide summary of the results, and discuss the potential impact on data analysis quality. Format your response in a professional, concise, and mature way, keeping it under 300 words, generate the response in markdown and don't include markdown heading in the response.
    """
    
    response = model.generate_content(prompt)
    return response.text

def generate_executive_summary(df, filtered_df, total_checks, passed_checks, quality_score):
    """Generate an AI-written executive summary based on dataset statistics."""
    
    # Calculate key metrics
    total_rows = len(df)
    total_columns = len(df.columns)
    missing_values = df.isnull().sum().sum()
    duplicate_rows = df.duplicated().sum()
    
    # Optional: Add more metrics if needed
    null_percentage = round((missing_values / (total_rows * total_columns)) * 100, 2) if total_rows and total_columns else 0

    prompt = f"""
    You are a data quality expert. Write a professional, concise, and insightful executive summary for a dataset with the following characteristics:
    
    - Total Rows: {total_rows}
    - Total Columns: {total_columns}
    - Missing Values: {missing_values}
    - Duplicate Rows: {duplicate_rows}
    - Null Percentage: {null_percentage}%
    
    Also include the following key recommendations:
    1. {filtered_df} 
    2. {total_checks} 
    3. {passed_checks}
    4. {quality_score}
    
    The summary should be under 300 words, written in a confident and analytical tone, and suitable for inclusion in a formal data quality report.
    The response should be in markown and don't include markdown heading in the response.
    """

    response = model.generate_content(prompt)
    return response.text


def generate_validation_insights(validation_data):
    """Generate AI insights from validation_data used in the PDF report."""
    # Skip header row
    data_rows = validation_data[1:]

    passed_columns = 0
    failed_columns = 0
    failed_details = []

    for row in data_rows:
        column_name = row[0]
        status = row[1]
        issues_paragraph = row[2]

        if "❌" in status:
            failed_columns += 1
            # Extract plain text from Paragraph object
            issue_text = issues_paragraph.getPlainText()
            failed_details.append(f"{column_name}: {issue_text}")
        else:
            passed_columns += 1

    failure_rate = (failed_columns / (passed_columns + failed_columns)) * 100 if (passed_columns + failed_columns) > 0 else 0
    failed_columns_list = ", ".join(failed_details[:5]) + ("..." if len(failed_details) > 5 else "")

    prompt = f"""
    You are a data quality expert. Write a professional, concise, and insightful summary of the validation results for a dataset.

    **Validation Overview**
    - {passed_columns} columns passed all validation checks.
    - {failed_columns} columns have issues that need attention.
    - Approx. {failure_rate:.1f}% of the columns failed at least one check.
    - Notable issues: {failed_columns_list}

    Keep the summary under 300 words. Use a confident, analytical tone suitable for a formal data quality report.
    The response should be in markdown format and don't include markdown heading in the response.
    """

    response = model.generate_content(prompt)
    return response.text



def generate_data_improvement_recommendations(
    df,
    schema,
    null_value_data,
    validation_data,
    total_checks,
    passed_checks,
    quality_score
):
    """Generate a final AI summary with improvement recommendations and next steps."""

    # Extract key metrics
    total_rows = len(df)
    total_columns = len(df.columns)
    missing_values = df.isnull().sum().sum()
    duplicate_rows = df.duplicated().sum()
    null_percentage = round((missing_values / (total_rows * total_columns)) * 100, 2) if total_rows and total_columns else 0

    # Count validation results
    data_rows = validation_data[1:]
    passed_columns = sum(1 for row in data_rows if "✅" in row[1])
    failed_columns = sum(1 for row in data_rows if "❌" in row[1])

    prompt = f"""
    You are a senior data quality consultant. Based on the following dataset profile, write a final summary that includes a holistic assessment, key findings, improvement recommendations, and next steps. The tone should be confident, professional, and forward-looking.

    **Dataset Overview**
    - Total Rows: {total_rows}
    - Total Columns: {total_columns}
    - Missing Values: {missing_values}
    - Duplicate Rows: {duplicate_rows}
    - Null Percentage: {null_percentage}%

    **Schema Overview**
    - Schema: {schema}

    **Validation Summary**
    - {passed_columns} columns passed all validation checks.
    - {failed_columns} columns have issues.
    - {passed_checks} out of {total_checks} validation checks passed.
    - Overall Quality Score: {quality_score:.1f}%

    **Null Value Insights**
    - {null_value_data}

    Provide:
    - A brief summary of the dataset's current state.
    - Key issues and their potential impact.
    - Actionable recommendations for improving data quality.
    - Suggested next steps for the data team.

    Keep the summary under 500 words.
    The response should be in markdown format and don't include markdown heading in the response.
    """

    response = model.generate_content(prompt)
    return response.text


#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def create_modern_pdf_report(df, validation_report, validation_rules, current_time, username):
    """Generate a modern PDF report with the required sections and embedded charts."""
    # We'll use reportlab to render a PDF
    
    output = io.BytesIO()
    doc = SimpleDocTemplate(
        output,
        pagesize=letter,
        rightMargin=0.5*inch,
        leftMargin=0.5*inch,
        topMargin=0.5*inch,
        bottomMargin=0.5*inch
    )
    
    # Create styles
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        spaceAfter=30,
        alignment=1,
        textColor=colors.HexColor('#1A5276')
    )
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=16,
        spaceBefore=20,
        spaceAfter=10,
        textColor=colors.HexColor('#2874A6')
    )
    subheading_style = ParagraphStyle(
        'CustomSubHeading',
        parent=styles['Heading3'],
        fontSize=14,
        spaceBefore=15,
        spaceAfter=8,
        textColor=colors.HexColor('#2E86C1')
    )
    normal_style = ParagraphStyle(
        'CustomNormal',
        parent=styles['Normal'],
        fontSize=11,
        spaceBefore=6,
        spaceAfter=6
    )
    insight_style = ParagraphStyle(
        'InsightStyle',
        parent=styles['Normal'],
        fontSize=11,
        spaceBefore=10,
        spaceAfter=10,
        leftIndent=20,
        rightIndent=20,
        borderColor=colors.HexColor('#AED6F1'),
        borderWidth=1,
        borderPadding=10,
        borderRadius=5
    )
    
    # Enhanced styles for AI insights
    insight_title_style = ParagraphStyle(
        'InsightTitleStyle',
        parent=styles['Heading3'],
        fontSize=14,
        spaceBefore=10,
        spaceAfter=5,
        textColor=colors.HexColor('#1A5276'),
        borderColor=colors.HexColor('#AED6F1'),
        borderWidth=0,
        borderPadding=0,
    )
    
    insight_bullet_style = ParagraphStyle(
        'InsightBulletStyle',
        parent=styles['Normal'],
        fontSize=11,
        spaceBefore=3,
        spaceAfter=3,
        leftIndent=30,
        firstLineIndent=-15,
        bulletIndent=15,
        bulletFontName='Symbol',
    )
    
    # Build document elements
    elements = []
    
    # Title
    elements.append(Paragraph("Data Validation Report", title_style))
    elements.append(Spacer(1, 20))
    
    # Metadata
    metadata = [
        ["Report Generated:", current_time],
        ["Generated By:", username],
        ["Total Records:", str(len(df))],
        ["Total Columns:", str(len(df.columns))]
    ]
    
    table = Table(metadata, colWidths=[2*inch, 4*inch])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#D4E6F1')),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 11),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
        ('TOPPADDING', (0, 0), (-1, -1), 12),
        ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#AED6F1'))
    ]))
    elements.append(table)
    elements.append(Spacer(1, 20))

    # 1. Executive Summary (SECTION 1)
    elements.append(Paragraph("1. Executive Summary", heading_style))
    
    # Calculate quality score using the new validation report format
    validation_columns = ['Type Valid', 'Range Valid', 'No Duplicates', 'Non-Null', 'Format Valid']
    total_checks = len(validation_columns) * len(validation_report)
    passed_checks = (validation_report[validation_columns] == '✅').sum().sum()
    quality_score = (passed_checks / total_checks) * 100 if total_checks > 0 else 0
    
    # Generate the summary using AI
    exec_summary_text = generate_executive_summary(df, validation_report, total_checks, passed_checks, quality_score)

    # Parse and format AI insights
    elements.append(Spacer(1, 10))
    elements.append(Paragraph("AI Insights - Executive Summary", insight_title_style))
    
    # Convert markdown to properly formatted bullets
    insight_elements = format_markdown_to_bullets(exec_summary_text, insight_bullet_style)
    elements.extend(insight_elements)
    
    # Add page break after executive summary
    elements.append(PageBreak())

    # 2. Data Preview Section (SECTION 2)
    elements.append(Paragraph("2. Data Preview", heading_style))
    
    # Data preview
    preview_data = [["Column", "Type", "Non-Null Count", "Sample Values"]]
    for col in df.columns:
        non_null_count = df[col].count()
        sample_vals = str(df[col].dropna().head(3).tolist())[:50] + "..."
        preview_data.append([
            col,
            str(df[col].dtype),
            f"{non_null_count} / {len(df)}",
            Paragraph(
                sample_vals,
                ParagraphStyle(
                    'SampleValueStyle',
                    fontSize=9,
                    fontName='Helvetica',
                    leading=12
                )
            )
        ])

    preview_table = Table(
        preview_data, 
        colWidths=[1.5*inch, inch, 1.5*inch, 3*inch],
        rowHeights=[0.4*inch] + [None] * (len(preview_data)-1)  # First row fixed, others auto
    )

    preview_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2874A6')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 11),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 11),
        ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#AED6F1')),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.whitesmoke, colors.HexColor('#EBF5FB')]),
        
        # Add these new styles for text wrapping
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
        ('RIGHTPADDING', (0, 0), (-1, -1), 6),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('WORDWRAP', (0, 0), (-1, -1), True)
    ]))
    elements.append(preview_table)
    
    # Add page break after data preview
    elements.append(PageBreak())
    
    # 3. Data Type Analysis Section (SECTION 3)
    elements.append(Paragraph("3. Data Type Analysis", heading_style))
    
    # Parse rules to display type comparison
    try:
        if hasattr(validation_rules, 'rules'):
            type_data = [["Column", "Expected Type", "Actual Type", "Type Match"]]
            
            for rule in validation_rules.rules:
                # Only include rules for columns that exist in the dataframe to remove empty rows
                if rule.column in df.columns and hasattr(rule, 'actual_type') and hasattr(rule, 'type_match'):
                    type_data.append([
                        rule.column,
                        rule.data_type,
                        getattr(rule, 'actual_type', 'Unknown'),
                        "✅" if getattr(rule, 'type_match', False) else "❌"
                    ])
            
            if len(type_data) > 1:  # If we have data
                type_table = Table(type_data, colWidths=[1.8*inch, 1.7*inch, 1.7*inch, 1.8*inch])
                type_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2874A6')),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 10),
                    ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
                    ('FONTSIZE', (0, 1), (-1, -1), 11),
                    ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#AED6F1')),
                    ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.whitesmoke, colors.HexColor('#EBF5FB')])
                ]))
                
                # Add colors for match column
                for i in range(1, len(type_data)):
                    if type_data[i][3] == "✅":
                        type_table.setStyle(TableStyle([
                            ('TEXTCOLOR', (3, i), (3, i), colors.green)
                        ]))
                    else:
                        type_table.setStyle(TableStyle([
                            ('TEXTCOLOR', (3, i), (3, i), colors.red)
                        ]))
                
                elements.append(type_table)
                
                # Generate AI insights for data types
                schema = {rule.column: rule.data_type for rule in validation_rules.rules if rule.column in df.columns}
                datatype_insights = generate_datatype_summaries(schema)

                # Format AI insights into bulletpoints
                elements.append(Spacer(1, 10))
                elements.append(Paragraph("AI Insights - Data Type Analysis", insight_title_style))
                
                insight_elements = format_markdown_to_bullets(datatype_insights, insight_bullet_style)
                elements.extend(insight_elements)

            else:
                elements.append(Paragraph("No data type analysis available.", normal_style))
        else:
            elements.append(Paragraph("No data type analysis available.", normal_style))
    except Exception as e:
        elements.append(Paragraph(f"Error displaying data type analysis: {str(e)}", normal_style))
    
    # Add page break after data type analysis
    elements.append(PageBreak())
    
    # 4. Null Value Distribution Section (SECTION 4)
    elements.append(Paragraph("4. Null Value Distribution", heading_style))
    
    # Create null values table
    null_data = [["Column", "Null Count", "Null Percentage", "Status"]]
    
    for col in df.columns:
        null_count = df[col].isnull().sum()
        null_percent = (null_count / len(df)) * 100
        
        if null_percent > 20:
            status = "High"
        elif null_percent > 5:
            status = "Medium"
        else:
            status = "Low"
        
        null_data.append([
            col,
            str(null_count),
            f"{null_percent:.2f}%",
            status
        ])
    
    null_table = Table(null_data, colWidths=[2*inch, 1.5*inch, 1.5*inch, 2*inch])
    null_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2874A6')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 11),
        ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#AED6F1')),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.whitesmoke, colors.HexColor('#EBF5FB')])
    ]))
    
    # Add colors for status column
    for i in range(1, len(null_data)):
        status = null_data[i][3]
        if status == "Low":
            null_table.setStyle(TableStyle([
                ('TEXTCOLOR', (3, i), (3, i), colors.green)
            ]))
        elif status == "Medium":
            null_table.setStyle(TableStyle([
                ('TEXTCOLOR', (3, i), (3, i), colors.orange)
            ]))
        elif status == "High":
            null_table.setStyle(TableStyle([
                ('TEXTCOLOR', (3, i), (3, i), colors.red)
            ]))
    
    elements.append(null_table)

    # Generate AI insights for null values
    null_value_insights = generate_nullvalue_summaries(null_data[1:]) # Exclude header row

    # Format AI insights into bulletpoints
    elements.append(Spacer(1, 10))
    elements.append(Paragraph("AI Insights - Null Value Analysis", insight_title_style))
    
    insight_elements = format_markdown_to_bullets(null_value_insights, insight_bullet_style)
    elements.extend(insight_elements)
    
    # Add page break after null value analysis
    elements.append(PageBreak())
    
    # 5. Validation Results Section (SECTION 5)
    elements.append(Paragraph("5. Validation Results", heading_style))
    validation_data = [[
        "Column", 
        "Type Valid",
        "Range Valid",
        "No Duplicates",
        "Non-Null",
        "Format Valid",
        "Issues"
    ]]

    for _, row in validation_report.iterrows():
        column_name = row['Column Name']
        
        if column_name in df.columns:
            validation_data.append([
                column_name,
                row['Type Valid'],
                row['Range Valid'],
                row['No Duplicates'],
                row['Non-Null'],
                row['Format Valid'],
                Paragraph(
                    row['Details'] if row['Details'] else "N/A",
                    ParagraphStyle(
                        'IssuesStyle',
                        fontSize=9,
                        fontName='Helvetica',
                        leading=12
                    )
                )
            ])

    validation_table = Table(
        validation_data, 
        colWidths=[1.2*inch, 0.8*inch, 0.8*inch, 0.8*inch, 0.8*inch, 0.8*inch, 1.8*inch],
        rowHeights=[0.4*inch] + [None] * (len(validation_data)-1)  # First row fixed, others auto
    )

    validation_table.setStyle(TableStyle([
        # Keep your existing styles
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2874A6')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 9),
        ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#AED6F1')),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.whitesmoke, colors.HexColor('#EBF5FB')]),
        
        # Add these new styles for text wrapping
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
        ('RIGHTPADDING', (0, 0), (-1, -1), 6),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('WORDWRAP', (0, 0), (-1, -1), True)
    ]))

    # Add colors for validation status columns
    for i in range(1, len(validation_data)):
        for col_idx in range(1, 6):  # Columns 1-5 are validation result columns
            if isinstance(validation_data[i][col_idx], str):
                status = validation_data[i][col_idx]
                if "✅" in status:
                    validation_table.setStyle(TableStyle([
                        ('TEXTCOLOR', (col_idx, i), (col_idx, i), colors.green)
                    ]))
                elif "❌" in status:
                    validation_table.setStyle(TableStyle([
                        ('TEXTCOLOR', (col_idx, i), (col_idx, i), colors.red)
                    ]))
    
    elements.append(validation_table)
    
    # Generate validation insights
    validation_insights = generate_validation_insights(validation_data)
    
    # Format AI insights into bulletpoints
    elements.append(Spacer(1, 10))
    elements.append(Paragraph("AI Insights - Validation Results", insight_title_style))
    
    insight_elements = format_markdown_to_bullets(validation_insights, insight_bullet_style)
    elements.extend(insight_elements)
    
    # Add page break after validation results
    elements.append(PageBreak())
    
    # 6. Data Validation Metrics (SECTION 6)
    elements.append(Paragraph("6. Data Validation Metrics", heading_style))
    
    # Create metrics table
    metrics_data = [
        ["Metric", "Value"],
        ["Overall Quality Score", f"{quality_score:.1f}%"],
        ["Checks Passed", f"{passed_checks}/{total_checks}"],
        ["Columns with Issues", str((validation_report[validation_columns] == '❌').any(axis=1).sum())],
        ["Completeness", f"{100 - (df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100):.1f}%"],
        ["Duplicate Rows", f"{df.duplicated().sum()} ({df.duplicated().sum() / len(df) * 100:.1f}%)"]
    ]
    
    metrics_table = Table(metrics_data, colWidths=[3*inch, 4*inch])
    metrics_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2874A6')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 9),
        ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#AED6F1')),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.whitesmoke, colors.HexColor('#EBF5FB')])
    ]))
    
    elements.append(metrics_table)
    
    # Generate AI insights for metrics
    metrics_insights = generate_metrics_summaries(validation_report, total_checks, passed_checks, quality_score)

    # Format AI insights into bulletpoints
    elements.append(Spacer(1, 10))
    elements.append(Paragraph("AI Insights - Data Quality Metrics", insight_title_style))
    
    insight_elements = format_markdown_to_bullets(metrics_insights, insight_bullet_style)
    elements.extend(insight_elements)
    
    # Add page break after metrics
    elements.append(PageBreak())
    
    # 7. Recommendations (NEW SECTION 7)
    elements.append(Paragraph("7. Data Improvement Recommendations", heading_style))
    
    # Generate AI recommendations
    recommendations = generate_data_improvement_recommendations(
        df=df,
        schema=schema,
        null_value_data=null_value_insights,
        validation_data=validation_data,
        total_checks=total_checks,
        passed_checks=passed_checks,
        quality_score=quality_score
    )
    
    # Format AI recommendations into bulletpoints
    elements.append(Spacer(1, 10))
    elements.append(Paragraph("AI Insights - Recommendations for Data Improvement", insight_title_style))
    
    recommendation_elements = format_markdown_to_bullets(recommendations, insight_bullet_style)
    elements.extend(recommendation_elements)
    
    doc.build(elements)
    
    return output


def format_markdown_to_bullets(markdown_text, bullet_style):
    """
    Converts markdown text into ReportLab bullet elements.
    
    Args:
        markdown_text (str): Markdown formatted text
        bullet_style (ParagraphStyle): The style to apply to bullet points
        
    Returns:
        list: List of Paragraph elements with bullet formatting
    """
    elements = []
    
    # Process the markdown text to extract sections
    sections = []
    current_section = {"title": None, "points": []}
    
    for line in markdown_text.split('\n'):
        line = line.strip()
        
        # Skip empty lines
        if not line:
            continue
            
        # Check if line is a heading (starts with # or has ** at beginning and end)
        if line.startswith('# ') or line.startswith('## ') or line.startswith('### '):
            # If we already have a section, add it to sections
            if current_section["points"]:
                sections.append(current_section)
                current_section = {"title": None, "points": []}
            
            # Set new section title
            current_section["title"] = line.lstrip('#').strip()
        
        # Check if line is a bold text (likely a section title)
        elif line.startswith('**') and line.endswith('**'):
            # If we already have a section, add it to sections
            if current_section["points"]:
                sections.append(current_section)
                current_section = {"title": None, "points": []}
            
            # Set new section title
            current_section["title"] = line.strip('*').strip()
        
        # Check if line is a bullet point
        elif line.startswith('- ') or line.startswith('* '):
            current_section["points"].append(line[2:].strip())
        
        # Otherwise treat as regular text, could be a continuation of previous bullet
        elif current_section["points"]:
            # Append to last bullet point
            current_section["points"][-1] += " " + line
        else:
            # Start a new bullet point
            current_section["points"].append(line)
    
    # Add the last section if it has content
    if current_section["points"]:
        sections.append(current_section)
    
    # Now build the elements
    for section in sections:
        if section["title"]:
            # Create section title with bold styling
            section_title_style = ParagraphStyle(
                'SectionTitle',
                parent=bullet_style,
                fontName='Helvetica-Bold',
                fontSize=14,
                spaceBefore=6,
                spaceAfter=3,
                leftIndent=10,
            )
            elements.append(Paragraph(section["title"], section_title_style))
        
        # Add bullet points
        for point in section["points"]:
            point = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', point)

            elements.append(Paragraph(f"• {point}", bullet_style))
    
    return elements


def generate_validation_summary(schema, validation_reports, combined_report_df, validation_columns):
    """Generate AI summary of validation logic, issues found, and remedies based on the dashboard's results."""
    
    # Extract key metrics
    total_rows = combined_report_df.shape[0]
    total_columns = combined_report_df.shape[1]
    failed_columns = (combined_report_df == '❌').any(axis=1).sum()
    # passed_checks = (combined_report_df == '✅').sum().sum()
    # total_checks = combined_report_df.size
    # quality_score = (passed_checks / total_checks) * 100 if total_checks > 0 else 0
    # print(quality_score)
    total_checks = len(validation_columns) * len(combined_report_df)
    passed_checks = (combined_report_df[validation_columns] == '✅').sum().sum()
    quality_score = (passed_checks / total_checks) * 100 if total_checks > 0 else 0

    # Build prompt
    prompt = f"""
You are a world best Data Scientist and Data Consultant Expert. Analyze the following validation results from a data quality assessment tool.

**Schema Summary**: {schema}

**Validation Reports**:
- Data Type Issues: {validation_reports['data_type'].to_dict(orient='records')}
- Range Issues: {validation_reports['range'].to_dict(orient='records')}
- Duplicates: {validation_reports['duplicate'].to_dict(orient='records')}
- Nulls: {validation_reports['null'].to_dict(orient='records')}
- Format Issues: {validation_reports['format'].to_dict(orient='records')}

**Overall Stats**:
- Total Rows: {total_rows}
- Total Columns: {total_columns}
- Columns with Issues: {failed_columns}
- Quality Score: {quality_score:.1f}%

Please:
1. Summarize what validation tests were performed to analyze the data.
2. Identify the types of problems found in the dataset.
3. Recommend practical solutions for each issue.

Respond in markdown (no headings), in a professional and concise tone.
    """

    response = model.generate_content(prompt)
    return response.text


def main():
    """
    Main function for the Data Validation Check App.
    Handles data loading, visualization, validation reporting, and PDF generation.
    """
    # Set current date and time dynamically
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    username = CURRENT_USER  # This could be retrieved from a login system in a production app
    
    # Configure the Streamlit page
    st.set_page_config(
        page_title="Data Quality Assessment App", 
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for a more modern look
    st.markdown("""
    <style>
    .main .block-container {padding-top: 2rem;}
    h1, h2, h3 {color: #1E88E5;}
    .stTabs [data-baseweb="tab-list"] {gap: 2px; background-color: #f0f2f6; border-radius: 4px;}
    .stTabs [data-baseweb="tab"] {padding: 10px 20px; border-radius: 4px 4px 0 0; background-color: #f0f2f6;}
    .stTabs [aria-selected="true"] {background-color: #1E88E5 !important; color: white !important;}
    .css-6qob1r {font-family: 'Segoe UI', sans-serif;}
    .css-1v0mbdj {border-radius: 5px;}
    .stDataFrame {border-radius: 8px; overflow: hidden;}
    .stMetric {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 5px;
        box-shadow: 0 1px 2px rgba(0,0,0,0.1);
        margin-bottom: 10px;
    }
    .stMetric:hover {
        background-color: #e8eaed;
    }
    /* Header styling for column names */
    h4 {
        color: #0e1117;
        margin-bottom: 10px;
        border-bottom: 2px solid #e6e6e6;
        padding-bottom: 5px;
    }
    /* Metric label colors */
    [data-testid="stMetricLabel"] {
        color: #666;
    }
    /* More modern metric styling */
    [data-testid="stMetricValue"] {font-size: 1.8rem !important; color: #1E88E5 !important; font-weight: bold !important;}
    [data-testid="stMetricLabel"] {font-size: 0.8rem !important; text-transform: uppercase !important; color: #555555 !important;}
    </style>
    """, unsafe_allow_html=True)
    
    # Create sidebar
    with st.sidebar:
        st.title("📊 Data Quality Assessment Tool")
        st.markdown("---")
        
        uploaded_file = st.file_uploader("Upload Dataset", type=['csv', 'json', 'xlsx', 'xls'])
        
        # Add app info
        st.sidebar.markdown("---")
        st.sidebar.markdown("### User Info")
        st.sidebar.markdown(f"**Time:** {current_time}")
        st.sidebar.markdown(f"**User:** {CURRENT_USER}")
    
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
                        # Calculate overall quality score
                        validation_columns = ['Type Valid', 'Range Valid', 'No Duplicates', 'Non-Null', 'Format Valid']
                        total_checks = len(validation_columns) * len(combined_report_df)
                        passed_checks = (combined_report_df[validation_columns] == '✅').sum().sum()
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
                                response = generate_validation_summary(schema, validation_reports, combined_report_df, validation_columns)
                                
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
                            # Create donut chart using plotly with increased height
                            fig = go.Figure(data=[go.Pie(
                                labels=['Pass', 'Fail'],
                                values=[passed_checks, total_checks - passed_checks],
                                hole=.6,
                                marker_colors=['#28a745', '#dc3545']
                            )])
                            
                            fig.update_layout(
                                annotations=[dict(text=f'{quality_score:.1f}%', x=0.5, y=0.5, font_size=20, showarrow=False)],
                                showlegend=True,
                                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
                                margin=dict(t=0, l=0, r=0, b=0),
                                height=250  # Increased height from 150 to 250
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)

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

                        # Calculate and display validation type summaries in a grid below both columns
                        validation_summaries = []
                        for validation_type in validation_columns:
                            passed = (combined_report_df[validation_type] == '✅').sum()
                            total = len(combined_report_df)
                            pass_rate = (passed / total) * 100
                            validation_summaries.append({
                                'type': validation_type,
                                'passed': passed,
                                'total': total,
                                'rate': pass_rate
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
                            # Create donut chart using plotly
                            fig = go.Figure(data=[go.Pie(
                                labels=['Pass', 'Fail'],
                                values=[matching_types, mismatched_types],
                                hole=.6,
                                marker_colors=['#28a745', '#dc3545']
                            )])
                            
                            fig.update_layout(
                                annotations=[dict(text=f'{match_percentage:.1f}%', x=0.5, y=0.5, font_size=20, showarrow=False)],
                                showlegend=True,
                                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
                                margin=dict(t=0, l=0, r=0, b=0),
                                height=200
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)

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
                            # Create donut chart using plotly
                            fig = go.Figure(data=[go.Pie(
                                labels=['Within Range', 'Out of Range'],
                                values=[in_range, out_range],
                                hole=.6,
                                marker_colors=['#28a745', '#dc3545']
                            )])
                            
                            fig.update_layout(
                                annotations=[dict(text=f'{in_range_percentage:.1f}%', x=0.5, y=0.5, font_size=20, showarrow=False)],
                                showlegend=True,
                                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
                                margin=dict(t=0, l=0, r=0, b=0),
                                height=200
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)

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
                            # Create donut chart using plotly
                            fig = go.Figure(data=[go.Pie(
                                labels=['Unique', 'Has Duplicates'],
                                values=[unique_cols, duplicate_cols],
                                hole=.6,
                                marker_colors=['#28a745', '#dc3545']
                            )])
                            
                            fig.update_layout(
                                annotations=[dict(text=f'{unique_percentage:.1f}%', x=0.5, y=0.5, font_size=20, showarrow=False)],
                                showlegend=True,
                                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
                                margin=dict(t=0, l=0, r=0, b=0),
                                height=200
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)

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
                            # Create donut chart using plotly
                            fig = go.Figure(data=[go.Pie(
                                labels=['No Nulls', 'Has Nulls'],
                                values=[no_nulls, has_nulls],
                                hole=.6,
                                marker_colors=['#28a745', '#dc3545']
                            )])
                            
                            fig.update_layout(
                                annotations=[dict(text=f'{no_nulls_percentage:.1f}%', x=0.5, y=0.5, font_size=20, showarrow=False)],
                                showlegend=True,
                                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
                                margin=dict(t=0, l=0, r=0, b=0),
                                height=200
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)

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
                            # Create donut chart using plotly
                            fig = go.Figure(data=[go.Pie(
                                labels=['Valid Format', 'Invalid Format'],
                                values=[valid_format, invalid_format],
                                hole=.6,
                                marker_colors=['#28a745', '#dc3545']
                            )])
                            
                            fig.update_layout(
                                annotations=[dict(text=f'{valid_percentage:.1f}%', x=0.5, y=0.5, font_size=20, showarrow=False)],
                                showlegend=True,
                                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
                                margin=dict(t=0, l=0, r=0, b=0),
                                height=200
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)

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

                # PDF Report section
                #st.header("Generate Report")
                
                # if st.button("Download PDF Report", type="primary"):
                #     with st.spinner("Generating comprehensive PDF report..."):
                #         try:
                #             # Create PDF in memory
                #             pdf_output = create_modern_pdf_report(
                #                 df, 
                #                 combined_report_df,
                #                 validation_rules,
                #                 current_time, 
                #                 CURRENT_USER
                #             )
                            
                #             # Add download button with timestamp in filename
                #             timestamp = datetime.strptime(current_time, "%Y-%m-%d %H:%M:%S")
                #             st.download_button(
                #                 label="📥 Download PDF Report",
                #                 data=pdf_output.getvalue(),
                #                 file_name=f"data_quality_report_{timestamp.strftime('%Y%m%d_%H%M%S')}.pdf",
                #                 mime="application/pdf",
                #             )
                            
                #             # Add generation info
                #             st.success(f"PDF Report successfully generated")
                #         except Exception as pdf_error:
                #             st.error(f"Error generating PDF report: {str(pdf_error)}")
        
        except Exception as e:
            st.error(f"Error processing dataset: {str(e)}")
            st.info("Please upload a valid dataset in CSV, JSON, or Excel format.")

if __name__ == "__main__":
    main()