# Run colab from your local VS-Code  

## Known issues  
The installation of a new environment is long > 10 minutes

## Instructions
- Open the auto-colab.ipynb file in goole colab  
- Run the first cell to generate the script to be later used to install conda on the Colab VM  
- Run the next cell to generate the env.yml file to be later used as a example of conda environment  
- Run the next cell to install VS-Code Server on the Colab VM and start it   
- Follow the instructions to set-up the tunneling from your local machine to the VS-Code Server  
- Skip the request to open the link in the browser  
- From your local Vs-Code, using the Remote-Tunnels extension from Microsoft, connect to the VS-Code Server  
- Run '. conda.sh' script to install conda and create a new environments  
- Create a notebook  
- To selec the environment ctrl+shift+p and select Python: Select Interpreter  