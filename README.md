# Data Quality Assessment Tool 📊

A comprehensive, AI-powered data validation and quality assessment application built with Streamlit and Google Gemini 2.5 Flash. This tool helps data scientists and analysts quickly identify data quality issues, generate validation reports, and receive actionable recommendations for data improvement.

## 🚀 Features

### Core Functionality
- **Multi-format Data Support**: Load CSV, JSON, and Excel files seamlessly
- **Comprehensive Data Validation**: 5 key validation categories
- **AI-Powered Insights**: Intelligent summaries and recommendations using Google Gemini 2.5 Flash
- **Interactive Dashboard**: Modern, responsive Streamlit interface
- **Professional PDF Reports**: Export detailed validation reports

### Validation Categories
1. **📊 Data Type Validation**: Verify data types match expected schemas
2. **📏 Range Validation**: Check numeric values against min/max constraints
3. **🔄 Duplicate Detection**: Identify duplicate values in unique constraint columns
4. **❓ Missing Value Analysis**: Comprehensive null value assessment
5. **🔤 Format Validation**: Regex pattern matching for data formats

### Advanced Features
- Real-time quality scoring with visual metrics
- Column-wise detailed analysis with interactive charts
- Executive summary with AI-generated insights
- Data improvement recommendations
- Professional PDF report generation with charts and insights

## 📋 Requirements

### Python Dependencies
```bash
pip install streamlit pandas matplotlib google-generativeai python-dotenv IPython reportlab plotly numpy markdown2 graphviz
```

### API Requirements
- **Google Gemini API Key**: Required for AI-powered insights and recommendations

## 🛠️ Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/rebel47/data-validation.git
   cd data-validation
   ```

2. **Install dependencies**
   ```bash
   pip install streamlit pandas matplotlib google-generativeai python-dotenv IPython reportlab plotly numpy markdown2 graphviz
   ```

3. **Set up environment variables**
   
   Create a `.env` file in the project root (adjust path as needed):
   ```bash
   # Create secrets.env or .env file
   GEMINI_API_KEY=your_gemini_api_key_here
   ```

4. **Run the application**
   ```bash
   streamlit run app.py
   ```

## 🔧 Configuration

### Environment Variables
Create a `secrets.env` file (or adjust the path in the code) with:

```env
GEMINI_API_KEY=your_google_gemini_api_key
```

### Getting a Gemini API Key
1. Visit [Google AI Studio](https://aistudio.google.com/)
2. Create an account or sign in
3. Generate a new API key
4. Add the key to your environment file

## 📖 Usage

### Getting Started
1. Launch the application using `streamlit run app.py`
2. Upload your dataset (CSV, JSON, or Excel format)
3. Review the data preview and basic statistics
4. Navigate through validation tabs to explore different quality checks
5. Generate and download comprehensive PDF reports

### Validation Workflow
1. **Data Preview**: Quick overview of your dataset structure
2. **Assessment Summary**: AI-powered overview with quality scoring
3. **Data Type Test**: Schema validation and type checking
4. **Range Test**: Numeric boundary validation
5. **Duplicates Test**: Uniqueness constraint verification
6. **Missing Values Test**: Null value analysis
7. **Format Test**: Pattern and format validation

### Generating Reports
- Click "Download PDF Report" to generate a comprehensive validation report
- Reports include executive summaries, detailed findings, and AI-generated recommendations
- All charts and metrics are embedded in the PDF

## 🏗️ Project Structure

```
data-validation/
├── app.py                 # Main Streamlit application
├── auth.py               # Authentication module (if applicable)
├── secrets.env           # Environment variables (create this)
├── requirements.txt      # Python dependencies (optional)
└── README.md            # This file
```

## 🎯 Use Cases

- **Data Scientists**: Validate datasets before model training
- **Data Engineers**: Quality checks in ETL pipelines
- **Business Analysts**: Ensure data integrity for reporting
- **Data Stewards**: Comprehensive data governance and quality monitoring
- **Teams**: Generate stakeholder-ready validation reports

## 🤖 AI Features

The application leverages Google Gemini 2.5 Flash to provide:
- **Executive Summaries**: High-level data quality assessments
- **Detailed Insights**: Column-specific analysis and recommendations
- **Improvement Recommendations**: Actionable steps to enhance data quality
- **Business Impact Analysis**: Understanding the implications of data issues

## 🔍 Example Validation Output

The tool provides detailed validation results including:
- ✅ **Pass/Fail Status**: Clear indicators for each validation check
- 📊 **Quality Scores**: Percentage-based quality metrics
- 📈 **Visual Charts**: Interactive pie charts and progress indicators
- 📝 **Detailed Reports**: Comprehensive analysis with AI insights
- 🎯 **Actionable Recommendations**: Specific steps to improve data quality

## 🚧 Development

### Contributing
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Code Structure
- **Data Models**: Custom classes replacing Pydantic models
- **Validation Engine**: Comprehensive rule-based validation system
- **AI Integration**: Google Gemini API for intelligent insights
- **Report Generation**: ReportLab-based PDF creation
- **UI Components**: Modern Streamlit interface with custom CSS

## 🐛 Troubleshooting

### Common Issues

**API Key Error**
```
Error: Google Gemini API key not found
Solution: Ensure GEMINI_API_KEY is set in your environment file
```

**File Upload Error**
```
Error: Unsupported file format
Solution: Use CSV, JSON, or Excel (.xlsx, .xls) files only
```

**Memory Issues with Large Files**
```
Solution: Consider chunking large datasets or increasing system memory
```

## 📊 Screenshots

*Add screenshots of your application here showing:*
- Main dashboard
- Validation results
- PDF report sample
- Charts and visualizations

## 🤝 Support

- **Issues**: [GitHub Issues](https://github.com/rebel47/data-validation/issues)
- **Discussions**: [GitHub Discussions](https://github.com/rebel47/data-validation/discussions)
- **Documentation**: This README and code comments

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Streamlit**: For the amazing web app framework
- **Google Gemini**: For powerful AI capabilities
- **ReportLab**: For PDF generation capabilities
- **Plotly**: For interactive visualizations
- **Pandas**: For data manipulation and analysis

## 🔗 Related Projects

- [Streamlit Documentation](https://docs.streamlit.io/)
- [Google Gemini API Documentation](https://ai.google.dev/)
- [Pandas Documentation](https://pandas.pydata.org/docs/)

---

**Built with ❤️ by [rebel47](https://github.com/rebel47)**

*If you find this project helpful, please consider giving it a ⭐ star on GitHub!*