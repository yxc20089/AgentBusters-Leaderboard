---
license: apache-2.0
task_categories:
- question-answering
language:
- zh
- en
pretty_name: e
configs:
  - config_name: cn
    data_files:
      - split: anomaly_information_tracing
        path: cn/anomaly_information_tracing_cn.jsonl
      - split: conterfactual
        path: cn/conterfactual_cn.jsonl
      - split: event_logic_reasoning
        path: cn/event_logic_reasoning_cn.jsonl
      - split: financial_data_description
        path: cn/financial_data_description_cn.jsonl
      - split: financial_multi_turn_perception
        path: cn/financial_multi-turn_perception_cn.jsonl
      - split: financial_quantitative_computation
        path: cn/financial_quantitative_computation_cn.jsonl
      - split: financial_report_analysis
        path: cn/financial_report_analysis.jsonl
      - split: stock_price_predict
        path: cn/stock_price_predict_cn.jsonl
      - split: user_sentiment_analysis
        path: cn/user_sentiment_analysis_cn.jsonl
  - config_name: en
    data_files:
      - split: anomaly_information_tracing
        path: en/anomaly_information_tracing_en.jsonl
      - split: conterfactual
        path: en/conterfactual_en.jsonl
      - split: event_logic_reasoning
        path: en/event_logic_reasoning_en.jsonl
      - split: financial_data_description
        path: en/financial_data_description_en.jsonl
      - split: financial_multi_turn_perception
        path: en/financial_multi-turn_perception_en.jsonl
      - split: financial_quantitative_computation
        path: en/financial_quantitative_computation_en.jsonl
      - split: stock_price_predict
        path: en/stock_price_predict_en.jsonl
      - split: user_sentiment_analysis
        path: en/user_sentiment_analysis_en.jsonl
---

<p align="center">
  <h1 align="center">BizFinBench.v2: A Unified Dual-Mode Bilingual Benchmark for Expert-Level Financial Capability Alignment</h1>
    <p align="center">
    <span class="author-block">
      Xin Guo<sup>1,2,*</sup> </a>,</span>
                <span class="author-block">
      Rongjunchen Zhang<sup>1,*,♠</sup></a>, Guilong Lu<sup>1</sup>, Xuntao Guo<sup>1</sup>, Jia Shuai<sup>1</sup>, Zhi Yang<sup>2</sup>, Liwen Zhang<sup>2,♠</sup>
    </span>
    </div>
    <div class="is-size-5 publication-authors" style="margin-top: 10px;">
        <span class="author-block">
            <sup>1</sup>HiThink Research, <sup>2</sup>Shanghai University of Finance and Economics
        </span>
        <br>
        <span class="author-block">
            <sup>*</sup>Co-first authors, <sup>♠</sup>Corresponding author, zhangrongjunchen@myhexin.com,zhang.liwen@shufe.edu.cn
        </span>
    </div>
  </p>
  <p>
  📖<a href="">Paper</a> |🏠<a href="">Homepage</a></h3>|🤗<a href="">Huggingface</a></h3>
  </p>
<div align="center"></div>
<p align="center">

**BizFinBench.v2** is the second release of [BizFinBench](https://github.com/HiThink-Research/BizFinBench). It is built entirely on real-world user queries from Chinese and U.S. equity markets. It bridges the gap between academic evaluation and actual financial operations.

<img src="static/score_sequence.png" alt="Evaluation Result">

### 🌟 Key Features

* **Authentic & Real-Time:** 100% derived from real financial platform queries, integrating online assessment capabilities.
* **Expert-Level Difficulty:** A challenging dataset of **29,578 Q&A pairs** requiring professional financial reasoning.
* **Comprehensive Coverage:** Spans **4 core business scenarios**, 8 fundamental tasks, and 2 online tasks.

### 📊 Key Findings
* **High Difficulty:** Even **ChatGPT-5** achieves only 61.5% accuracy on main tasks, highlighting a significant gap vs. human experts.
* **Online Prowess:** **DeepSeek-R1** outperforms all other commercial LLMs in dynamic online tasks, achieving a total return of 13.46% with a maximum drawdown of -8%.

## 📢 News 
- 🚀 [06/01/2026] TBD

## 📕 Data Distribution
BizFinBench.v2 contains multiple subtasks, each focusing on a different financial understanding and reasoning ability, as follows:

### Distribution Visualization
<div align="center">
    <img src="static/distribution.png" alt="Data Distribution" width="600">
</div>

### Detailed Statistics
| Scenarios | Tasks | Avg. Input Tokens | # Questions |
|:---|:---|---:|---:|
| **Business Information Provenance** | Anomaly Information Tracing | 8,679 | 4,000 |
| | Financial Multi-turn Perception | 10,361 | 3,741 |
| | Financial Data Description | 3,577 | 3,837 |
| **Financial Logic Reasoning** | Financial Quantitative Computation | 1,984 | 2,000 |
| | Event Logic Reasoning | 437 | 4,000 |
| | Counterfactual Inference | 2,267 | 2,000 |
| **Stakeholder Feature Perception** | User Sentiment Analysis | 3,326 | 4,000 |
| | Financial Report Analysis | 19,681 | 2,000 |
| **Real-time Market Discernment** | Stock Price Prediction | 5,510 | 4,000 |
| | Portfolio Asset Allocation | — | — |
| **Total** | **—** | **—** | **29,578** |


## ✒️Citation

```
Coming Soon
```

## 📄 License
![Code License](https://img.shields.io/badge/Code%20License-Apache_2.0-green.svg) ![Data License](https://img.shields.io/badge/Data%20License-CC%20By%20NC%204.0-red.svg) **Usage and License Notices**: The data and code are intended and licensed for research use only.
License: Attribution-NonCommercial 4.0 International It should abide by the policy of OpenAI: https://openai.com/policies/terms-of-use

## 💖 Acknowledgement
* Special thanks to Ning Zhang, Siqi Wei, Kai Xiong, Kun Chen and colleagues at HiThink Research's data team for their support in building BizFinBench.v2.

