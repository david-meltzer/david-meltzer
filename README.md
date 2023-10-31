# Welcome

Hi, my name is David Meltzer and I am a postdoctoral researcher in theoretical physics at Cornell University. 

My research work in physics was primarily on the study of formal aspects of Quantum Field Theory (QFT) and the AdS/CFT correspondence. Currently, I am interested in various aspects of Machine Learning (ML), including Natural Language Processing, Computer Vision, and how lessons from physics can be used to understand large neural nets. For more information, please see my Linkedin profile <a href="https://www.linkedin.com/in/david-meltzer-12a72162/">here</a>. A full list of my research publications can be found on <a href="https://scholar.google.com/citations?hl=en&user=jkez7jMAAAAJ">Google scholar</a> or on <a href="https://inspirehep.net/authors/1430989?ui-citation-summary=true">InspireHEP</a>.

# Projects

I have completed a few projects in machine learning which I will summarize below, with additional links for more details.

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
