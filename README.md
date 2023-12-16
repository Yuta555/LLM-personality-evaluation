# Do LLMs have personalities

### Group members Name UNI 
- Mouwei Lin lm3756 (Team captain)
- Frank Li zl3204
- Linhao Yu ly2590
- Yuta Adachi ya2488
- Xingye Feng xf2248

Emails  &lt;UNI&gt; @ columbia.edu

**JPMorgan mentor & co-mentors:** Akshat Gupta

**Instructor/CA:** Jiaxuan Li

The evolution of Language Learning Models (LLMs) like ChatGPT represents a significant paradigm shift in human-machine interactions. These advanced LLMs not only process information and generate outputs but also simulate interactions with nuances akin to human ’personality traits’. Understanding, controlling, and potentially modifying these traits is at the heart of this research.


**Directory tree**
```
│   .gitignore
│   CONTRIBUTING.md
│   README.md
│   Personality_Test_OCEAN.ipynb
│   .DS_Store
│
├───data_preprocess
│   ├───data
│   │       category.pkl
│   │       news.xlsx
│   │       summary.pkl
│   │       tweets.pkl
│   │       with_replacement_100.csv
│   │       without_replacement_100.csv
│   │
│   └───.ipynb_checkpoints
│           LLama2_summarizer.ipynb
│           newsscrap.ipynb
│           tweet_personality_detection.ipynb
│           news_summary_and_category_and_generate_tweets.ipynb
├───dual_method_self_test
│        Personality_Test_OCEAN.ipynb
│       
├───personality_detection_model
│   │   prediction_template.ipynb
│   │   README.md
│   │   requirements.txt
│   │
│   ├───data
│   │   │   kaggle.csv
│   │   │
│   │   └───processed_data_50tweets
│   │       │   dataset_dict.json
│   │       │  
│   │       ├───test
│   │       │       cache-0f9c89e60daef715.arrow
│   │       │       data-00000-of-00001.arrow
│   │       │       dataset_info.json
│   │       │       state.json
│   │       │
│   │       └───train
│   │               cache-77b616e2fc9aba67.arrow
│   │               data-00000-of-00001.arrow
│   │               dataset_info.json
│   │               state.json
│   │      
│   ├───docs      
│   │       Manual_for_multiGPU_setup.md
│   │
│   ├───previous_paper
│   │       replicate_previous_paper.ipynb
│   │
│   ├───results
│   │   │   aggregation.ipynb
│   │   │   results_on_testdata.csv
│   │   │
│   │   ├───binary_cls
│   │   │       binary_to_multi.ipynb
│   │   │       test_results_binary.json
│   │   │       test_results_dim1.json
│   │   │       test_results_dim2.json
│   │   │       test_results_dim3.json
│   │   │       test_results_dim4.json
│   │   │
│   │   └───multi_cls
│   │           multi_to_binary.ipynb
│   │           test_results_multi.json
│   │   
│   └───scripts
│           evaluation.py
│           finetuning.py
│           preprocess_data.py
│
└───self_assessment_test
        llms_mbti_en.json
        mbti_questions_en.json
        MBTI_translation.ipynb
        self-assessment_test.ipynb
        translation_results.csv
```



