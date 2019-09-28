##IMPORTANT : Run the code using the Bash command not the sh command .So to run the code do the following: bash runmanycode.sh
##This is a shell script to run multiple instance of K and n 
##Step 1 : Create the folder where the output is going to be : 
##Step2 : Remove the replication folder if already exists 
##Step 3 : Run the python Script


for ((K=1;K<=300;K=K+5))
do
  mkdir "/home/abhishek/Desktop/V2_MNIST_$K""_5"
  rm -rf "/home/abhishek/Desktop/ANU/comp_6470/adapted_deep_embeddings/trained_models/mnist/mnist_$K""_5/weight_transfer/replication1"
python3 main.py @"trained_models/mnist/mnist_$K""_5/weight_transfer/opts.txt"

done

#mkdir /home/abhishek/Desktop/V2_MNIST_1_5 




#rm -rf /home/abhishek/Desktop/ANU/comp_6470/adapted_deep_embeddings/#trained_models/#mnist/mnist_1_5/weight_transfer/replication1


#python3 main.py @trained_models/mnist/mnist_1_5/weight_transfer/opts.txt


#mkdir /home/abhishek/Desktop/V2_MNIST_2_5 
#rm -rf /home/abhishek/Desktop/ANU/comp_6470/adapted_deep_embeddings/#trained_models/mnist/mnist_2_5/weight_transfer/replication1
#python3 main.py @trained_models/mnist/mnist_2_5/weight_transfer/opts.txt




