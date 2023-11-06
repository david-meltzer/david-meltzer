# Welcome

Hi, my name is David Meltzer and I am a postdoctoral researcher in theoretical physics at Cornell University. 

My research work in physics was primarily on the study of formal aspects of Quantum Field Theory (QFT) and the AdS/CFT correspondence. Currently, I am interested in various aspects of Machine Learning (ML), including Natural Language Processing, Computer Vision, and how lessons from physics can be used to understand large neural nets. For more information, please see my Linkedin profile <a href="https://www.linkedin.com/in/david-meltzer-12a72162/">here</a>. A full list of my research publications can be found on <a href="https://scholar.google.com/citations?hl=en&user=jkez7jMAAAAJ">Google scholar</a> or on <a href="https://inspirehep.net/authors/1430989?ui-citation-summary=true">InspireHEP</a>. 

Please feel free to reach out to me on Linkedin if you have any questions about my work or have any issues accessing the reports and applications which are detailed below.

# Projects

I have completed a few projects in machine learning which I will summarize below, with additional links for more details. As a disclaimer, the applications may need to be restarted on Hugging Face if they have been inactive for too long.

<a href="https://github.com/david-meltzer/Longform-Question-Generation">**Scientific Question Generation**</a>
<ul>
 <li>Here I looked at training encoder-decoder transformer models to produce questions from scientific text. This project was inspired by this <a href="https://arxiv.org/abs/2210.11536">paper</a>.</li>
 <li>I wrote two Weights and Biases reports (see <a href="https://api.wandb.ai/links/dmeltzer/7an677es">1</a> and <a href="https://wandb.ai/dmeltzer/Question_Generation/reports/Exploratory-Data-Analysis-for-r-AskScience--Vmlldzo0MjQwODg1">2</a>) where I performed exploratory data analysis for the dataset, explained the architecture of the models and how they are trained, and compared the performance of the fine-tuned models with the zero-shot performance of larger foundation models, such as GPT-3.5 and FLAN-T5-XXL.
 </li>
 <li>
  I wrote two Streamlit applications which are hosted on Huggingface. The first application <a href="https://huggingface.co/spaces/dhmeltzer/qg_generation">generates questions</a> using both the fine-tuned BART-large models as well as the GPT-3.5 and FLAN-T5-XXL models. The fine-tuned models are deployed on AWS. The second application allows users to perform a <a href="https://huggingface.co/spaces/dhmeltzer/semantic">semantic search</a> over the dataset.
 </li>
</ul>

<a href="https://github.com/david-meltzer/Goodreads-Sentiment-Analysis">**Sentiment Analysis For Book Reviews**</a>
<ul>
 <li>In this project I used two encoder only models, BERT-tiny and distilBERT to predict the rating of a review. That is, given the text of a review can we predict the amount of stars assigned by the user?</li>
 <li> I wrote three Weights and Biases reports which can be found at the following links: <a href="https://api.wandb.ai/links/dmeltzer/ilnx2o0v">1</a>, <a href="https://api.wandb.ai/links/dmeltzer/s840cljt">2</a>, <a href="https://wandb.ai/dmeltzer/mlops-course-assgn3/reports/Goodreads-Reviews-Week-3--VmlldzozNzYxODkz">3</a>. In these reports I performed exploratory data analysis, compared the performance of the two models, and studied why the models' predictions differed from the ground-truth labels.</li>
 <li> I also wrote an application which <a href="https://huggingface.co/spaces/dhmeltzer/Sentiment-of-Book-Reviews">predicts the review rating</a> for a user-inputted review. The models used in this application are again deployed on AWS.</li>
</ul>

<a href="https://github.com/david-meltzer/gutenberg">**Visualizing Literature with Transformers**</a>
<ul>
 <li>Here I used SentenceTransformer models to visualize different works of literature. Specifically, I looked at visualizing three books by James Joyce, "Dubliners", "Portrait of the Artist as a Young Man" and "Ulysses" as well as three different translations of Homer's "The Odyssey".</li>
 <li>I embedded sentences from these books into a high-dimensional vector space and visualized how sentences from different chapters are related using a heat-map and the dimensionality-reduction methods, t-SNE and UMAP. These plots are interactive so users can see which chapters are "closest" to each other. </li>
 <li>The results of this analysis is summarized in the following Weights and Biases <a href="https://wandb.ai/dmeltzer/gutenberg/reports/Copy-of-Visualizing-Literature-using-Transformers--Vmlldzo1NzU5NjU0">report</a>, where I also reviewed the training procedure for SentenceTransformer models. </li>
</ul>

<a href="https://github.com/david-meltzer/quadratic_model">**Quadratic Models and the Catapult Mechanism**</a>
<ul>
 <li>This repository contains the code used in the following research <a href="https://arxiv.org/abs/2301.07737">paper</a>. This work directly builds off of the original paper on the <a href="https://arxiv.org/abs/2003.02218">catapult mechanism</a>.</li>
 <li>In the above paper, we gave sufficient conditions for when quadratic models and two-layer, scale-invariant neural networks exhibit the catapult mechanism, which is a non-perturbative, dynamical phase transition of wide neural nets. In the process, we demonstrated how large learning rates can lead to an implicit $L_2$ regularization of neural nets.</li>
 <li>We also experimentally demonstrated that training ReLU nets with a large learning rate causes their activation map to become sparse. We conjecture that this sparsity is also important in explaining why increasing the learning rate causes the generalization, or test, loss to decrease.</li>
</ul>

<a href="https://github.com/david-meltzer/LLMs">**Training a Llama-2 Model on Safe Simple Text**</a>
<ul>
 <li>
  This project is a work-in-progress and will be continually updated as we get new results. The main goal of this work is to train Llama-2 models such that they answer questions in a safe and simple way. That is, the model should not produce toxic content and its explanation should be understandable to as many people as possible.
 </li>
 <li>
  To accomplish this, we train Llama-2 models with supervised fine-tuning on question/answer posts that have been filtered for simplicity using Flesch-Kincaid readability metrics and for toxicity using a fine-tuned RoBERTa model. Given the size of the model, we train it using QLORA, which is a form of parameter-efficient fine-tuning (PEFT). In practice, we form the dataset by either generating synthetic questions from Wikipedia articles using GPT-3.5 and by filtering an existing dataset of Reddit posts.
 </li>
 <li>Our current results are summarized in a repot which can be downloaded <a href="https://github.com/david-meltzer/LLMs/blob/main/report/summary.pdf">here</a>. We observe that the fine-tuned models perform competitively we similarly sized models on the Huggingface <a="https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard">leaderboard</a> and that the fine-tuned models do produce simpler text according to our readability metrics.</li>
 <li> In the future, we will further explore how to train QA models such that they are aligned with human preferences, e.g. by using reinforcement learning from human feedback (RLHF) and direct preference optimization (DPO).</li>
</ul>

# Courses

Some of my Github repositories contain solutions to courses I have completed. 

<a href="https://github.com/david-meltzer/ML_finance">**ML Finance**</a>

<ul>
 <li>In this repository you can find solutions to some assignments of a course on <a href="https://www.coursera.org/learn/reinforcement-learning-in-finance?specialization=machine-learning-reinforcement-finance">Reinforcement Learning in Finance</a>. Specifically, I included the solutions to the assignments specifically focused on using reinforcement learning to price European options.</li>
 <li>This repository also contains some personal projects, looking at topics such as clustering, pair trading, and eigenportfolios for US equities. These assignments are inspired from this nice <a href="https://www.amazon.com/Machine-Learning-Algorithmic-Trading-alternative/dp/1839217715">textbook</a> by Stefan Jansen.</li>
 <li>Finally, you can also find a notebook studying credit card fraud using various classification algorithms, including logistic regression, random forests, and Gaussian Naive Bayes. The original credit dataset can be found on <a href="https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud">Kaggle</a>.</li>
</ul>

**Stanford Courses: <a href="https://github.com/david-meltzer/CS_224n">CS224n</a> and <a href="https://github.com/david-meltzer/CS_231n">CS 231n</a>**
<ul>
 <li>
  These two repositories contain the solutions to assignments from Stanford's computer science courses on natural language processing (CS224n) and computer vision (CS231n).
 </li>
</ul>
 
<!--
**david-meltzer/david-meltzer** is a ✨ _special_ ✨ repository because its `README.md` (this file) appears on your GitHub profile.

Here are some ideas to get you started:

- 🔭 I’m currently working on ...
- 🌱 I’m currently learning ...
- 👯 I’m looking to collaborate on ...
- 🤔 I’m looking for help with ...
- 💬 Ask me about ...
- 📫 How to reach me: ...
- 😄 Pronouns: ...
- ⚡ Fun fact: ...
-->
