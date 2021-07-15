# Entity Extraction Using BERT
Extract different entities present in a sentence.<br><br>
More about BERT can be found [here](https://huggingface.co/bert-base-uncased)

- End to End NLP Entity Extraction project.
- The Kaggle dataset can be found Here [Click Here](https://www.kaggle.com/abhinavwalia95/entity-annotated-corpus)
- My kaggle Notebook can be found [here](https://www.kaggle.com/raryan/entity-extraction-bert)
 
## Steps to Run the Project:
- create a virtual environment and install requirements.txt
  
### For Training
- After Setting up the environment go to **backend/services/entity_extraction/application/ai/training/** and run **main.py** and the training will start.
- After training is complete the weights of the model will be saved in weights directory, and this weights can be used for inference.
  
### For Prediction/Inference
- Download the pre-trained weights from [here](https://drive.google.com/file/d/1ytF8UWUJ_DYmRy57Iy5NjgUXYfTMEW-D/view?usp=sharing).
- After downloading the .bin(weights) file, place it inside (**backend/services/entity_extraction/application/ai/weights/**) directory.
- After setting up the environment: go to **backend/services/entity_extraction/api** and run **app.py**.
- After running the above step the server will start.  
- You can send the POST request at this URL - **localhost:8080/entity_extraction/api/v1/predict** 
- you can find the declaration of endpoint under **backend/services/toxic_comment_jigsaw/api/__init__.py**
- You can also see the log under(**backend/services/entity_extraction/logs**) directory.

Following are the screenshots for the output, and the request.

- Request sample 
![Sample request](https://github.com/R-aryan/Entity-Extraction-Bert/blob/develop/msc/sample_request.png)
  <br>
  <br>
- Response Sample
![Sample response_1](https://github.com/R-aryan/Entity-Extraction-Bert/blob/develop/msc/sample_response_1.png)
  <br>
  <br>
![Sample response_2](https://github.com/R-aryan/Entity-Extraction-Bert/blob/develop/msc/sample_response_2.png)