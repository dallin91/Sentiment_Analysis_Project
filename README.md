<h1>Sentiment Analysis with RoBERTa</h1>

<p>This project utilizes the RoBERTa model to perform sentiment analysis on open-ended responses from Excel files. The goal is to classify the responses into three categories: positive, negative, or neutral.</p>

<h2>Table of Contents</h2>

<ul>
  <li><a href="#installation">Installation</a></li>
  <li><a href="#usage">Usage</a></li>
  <li><a href="#data-format">Data Format</a></li>
  <li><a href="#model-training">Model Training</a></li>
  <li><a href="#evaluation">Evaluation</a></li>
  <li><a href="#contributing">Contributing</a></li>
  <li><a href="#license">License</a></li>
</ul>

<h2 id="installation">Installation</h2>

<p>To install the required dependencies and set up the project, follow these steps:</p>

<ol>
  <li>Clone the repository: <code>git clone https://github.com/dallin91/Sentiment_Analysis_Project.git</code></li>
  <li>Change to the project directory: <code>cd sentiment-analysis</code></li>
  <li>Create a virtual environment: <code>python3 -m venv env</code></li>
  <li>Activate the virtual environment:
    <ul>
      <li>On Windows: <code>env\Scripts\activate.bat</code></li>
      <li>On macOS and Linux: <code>source env/bin/activate</code></li>
    </ul>
  </li>
  <li>Install the dependencies: <code>pip install -r requirements.txt</code></li>
</ol>

<h2 id="usage">Usage</h2>

<p>To use the sentiment analysis project, follow these steps:</p>

<ol>
  <li>Prepare your input data in an Excel file (refer to the <a href="#data-format">Data Format</a> section for details).</li>
  <li>Run the sentiment analysis script: <code>python analyze_sentiment.py --input-file path/to/input.xlsx</code></li>
  <li>The script will process the responses and output the sentiment analysis results.</li>
</ol>

<h2 id="data-format">Data Format</h2>

<p>The input data should be provided in an Excel file with the following format:</p>

<table>
  <thead>
    <tr>
      <th>ID</th>
      <th>Response Text</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>1</td>
      <td>This is a positive response.</td>
    </tr>
    <tr>
      <td>2</td>
      <td>I am not happy with the product.</td>
    </tr>
    <tr>
      <td>3</td>
      <td>The service was neither good nor bad.</td>
    </tr>
    <!-- Add more rows as needed -->
  </tbody>
</table>

<p>Ensure that the Excel file contains a column named "Response Text" that contains the open-ended responses. The "ID" column is optional but can be useful for reference.</p>

<h2 id="model-training">Model Training</h2>

<p>This project utilizes the RoBERTa model, which has already been pre-trained on a large corpus. If you wish to fine-tune the model on your own dataset, you can refer to the <a href="model_training.md">model_training.md</a> file for detailed instructions.</p>

<h2 id="evaluation">Evaluation</h2>

<p>To evaluate the performance of the sentiment analysis model, refer to the <a href="evaluation.md">evaluation.md</a> file for guidance on assessing accuracy, precision, recall, and other relevant metrics.</p>

<h2 id="contributing">Contributing</h2>

<p>Contributions to this project are welcome! If you find any issues or would like to suggest improvements, please feel free to submit a pull request or open an issue on the GitHub repository.</p>

<h2 id="license">License</h2>

<p>This project is licensed under the <a href="LICENSE">MIT License</a>. You are free to modify and use the code for both commercial and non-commercial purposes.</p>
