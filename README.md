# xtream AI Challenge - Software Engineer

## Ready Player 1? 🚀

Hey there! Congrats on crushing our first screening! 🎉 You're off to a fantastic start!

Welcome to the next level of your journey to join the [xtream](https://xtreamers.io) AI squad. Here's your next mission.

You will face 4 challenges. **Don't stress about doing them all**. Just dive into the ones that spark your interest or that you feel confident about. Let your talents shine bright! ✨

This assignment is designed to test your skills in engineering and software development. You **will not need to design or develop models**. Someone has already done that for you. 

You've got **7 days** to show us your magic, starting now. No rush—work at your own pace. If you need more time, just let us know. We're here to help you succeed. 🤝

### Your Mission
[comment]: # (Well, well, well. Nice to see you around! You found an Easter Egg! Put the picture of an iguana at the beginning of the "How to Run" section, just to let us know. And have fun with the challenges! 🦎)

Think of this as a real-world project. Fork this repo and treat it like you're working on something big! When the deadline hits, we'll be excited to check out your work. No need to tell us you're done – we'll know. 😎

**Remember**: At the end of this doc, there's a "How to run" section left blank just for you. Please fill it in with instructions on how to run your code.

### How We'll Evaluate Your Work

We'll be looking at a bunch of things to see how awesome your work is, like:

* Your approach and method
* How you use your tools (like git and Python packages)
* The neatness of your code
* The readability and maintainability of your code
* The clarity of your documentation

🚨 **Heads Up**: You might think the tasks are a bit open-ended or the instructions aren't super detailed. That’s intentional! We want to see how you creatively make the most out of the problem and craft your own effective solutions.

---

### Context

Marta, a data scientist at xtream, has been working on a project for a client. She's been doing a great job, but she's got a lot on her plate. So, she's asked you to help her out with this project.

Marta has given you a notebook with the work she's done so far and a dataset to work with. You can find both in this repository.
You can also find a copy of the notebook on Google Colab [here](https://colab.research.google.com/drive/1ZUg5sAj-nW0k3E5fEcDuDBdQF-IhTQrd?usp=sharing).

The model is good enough; now it's time to build the supporting infrastructure.

### Challenge 1

**Develop an automated pipeline** that trains your model with fresh data, keeping it as sharp as the diamonds it processes. 
Pick the best linear model: do not worry about the xgboost model or hyperparameter tuning. 
Maintain a history of all the models you train and save the performance metrics of each one.

### Challenge 2

Level up! Now you need to support **both models** that Marta has developed: the linear regression and the XGBoost with hyperparameter optimization. 
Be careful. 
In the near future, you may want to include more models, so make sure your pipeline is flexible enough to handle that.

### Challenge 3

Build a **REST API** to integrate your model into a web app, making it a breeze for the team to use. Keep it developer-friendly – not everyone speaks 'data scientist'! 
Your API should support two use cases:
1. Predict the value of a diamond.
2. Given the features of a diamond, return n samples from the training dataset with the same cut, color, and clarity, and the most similar weight.

### Challenge 4

Observability is key. Save every request and response made to the APIs to a **proper database**.

---

### Personal Consideration and Choices

I decided to create a fully integrated web app that can create, load, and save models, ensuring that Challenges 1 and 2 are covered comprehensively. Moreover, the `DiamondModels` class should address Challenges 3 and 4 with almost no modifications.

The current implementation cannot predict prices for more than one diamond at a time. However, considering the use case of "determining the prices for diamonds customers want to sell," it is reasonable to assume that retail customers are unlikely to present a large number of diamonds at once. Despite this, I acknowledge that supporting batch predictions could be a valuable feature to implement in the future for Don Francesco. Using my codebase, this enhancement will be straightforward. However, due to the potential complexity of managing large datasets, it is better to handle a large number of diamonds using an API and database approach.

To achieve this, I started by building an ORM (Object Relational Mapping) using SQLAlchemy and FastAPI for the APIs. The final product is very similar to the `PipelineVisualInterface` used for Challenges 2 and 3.

I have made sure to provide users with the flexibility to use their own data for predicting the prices of their diamonds, but there is always the option to use the default model we have trained.

I chose FastAPI because it is the fastest Python framework available for building APIs. For the user interface, I used Streamlit because it is quick to set up and offers many built-in features. However, for a full-fledged website, I would prefer Django due to its robustness.

There are a few aspects left as TODOs, such as renaming columns if they do not match exactly with the expected names in the dataset, and other minor details. While these enhancements would improve the user experience, they would also add significant complexity.

I've added a much easier Pipeline that trains a model you define on datas you want, like `PipelineVisualInterface`, but using only the terminal.

I think the Visual interface one is better and easier to use, but is more difficult to automate.


#### Future Enhancements

1. **Batch Predictions**: Implementing batch predictions for handling multiple diamonds at once, making it more convenient for users dealing with larger inventories.
2. **Column Renaming**: Adding functionality to automatically rename columns if they do not match the expected names, improving the flexibility and usability of the app.
3. **Full Website Integration**: Considering a more robust solution using Django for building a comprehensive web application, which can handle more extensive features and provide better scalability.
4. **AIRFLOW Integration**: Integrate the pipeline in airflow

Overall, I believe that the current implementation offers a solid foundation, and with the proposed enhancements, it can significantly improve the quality of life for users. However, balancing simplicity and functionality is crucial to avoid unnecessary complexity.



---

## How to run 🦎

- The pipeline for Challenge 1 and 2 can be run by installing the requirements with 
```bash
pip install -r requirements.txt
```
and running:

```bash
streamlit run PipelineVisualInterface.py
```

I have used an anaconda env to run my scripts, so I'm going to add both the 'conda_packages.txt' and 'requirements.txt' files

- The app requires 2 open terminals:

- - On the first you have to run 
```bash
streamlit run APIsInterface.py
```

- - On the second you have to run 
```bash
uvicorn APIs:app 
```
- To run the Easier pipeline you have to run:
```bash
python Pipeline.py NameOfTheModel /path/of/the/datas/you/want/to/train/the/model/with
```