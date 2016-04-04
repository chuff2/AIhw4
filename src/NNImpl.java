/**
 * The main class that handles the entire network
 * Has multiple attributes each with its own use
 * 
 */

import java.util.*;


public class NNImpl{
	public ArrayList<Node> inputNodes=null;//list of the output layer nodes.
	public ArrayList<Node> hiddenNodes=null;//list of the hidden layer nodes
	public ArrayList<Node> outputNodes=null;// list of the output layer nodes
	
	public ArrayList<Instance> trainingSet=null;//the training set
	
	Double learningRate=1.0; // variable to store the learning rate
	int maxEpoch=1; // variable to store the maximum number of epochs
	
	/**
 	* This constructor creates the nodes necessary for the neural network
 	* Also connects the nodes of different layers
 	* After calling the constructor the last node of both inputNodes and  
 	* hiddenNodes will be bias nodes. 
 	*/
	
	public NNImpl(ArrayList<Instance> trainingSet, int hiddenNodeCount, Double learningRate, int maxEpoch, Double [][]hiddenWeights, Double[][] outputWeights)
	{
		this.trainingSet=trainingSet;
		this.learningRate=learningRate;
		this.maxEpoch=maxEpoch;
		
		//input layer nodes
		inputNodes=new ArrayList<Node>();
		int inputNodeCount=trainingSet.get(0).attributes.size();
		int outputNodeCount=trainingSet.get(0).classValues.size();
		for(int i=0;i<inputNodeCount;i++)
		{
			Node node=new Node(0);
			inputNodes.add(node);
		}
		
		//bias node from input layer to hidden
		Node biasToHidden=new Node(1);
		inputNodes.add(biasToHidden);
		
		//hidden layer nodes
		hiddenNodes=new ArrayList<Node> ();
		for(int i=0;i<hiddenNodeCount;i++)
		{
			Node node=new Node(2);
			//Connecting hidden layer nodes with input layer nodes
			for(int j=0;j<inputNodes.size();j++)
			{
				NodeWeightPair nwp=new NodeWeightPair(inputNodes.get(j),hiddenWeights[i][j]);
				node.parents.add(nwp);
			}
			hiddenNodes.add(node);
		}
		
		//bias node from hidden layer to output
		Node biasToOutput=new Node(3);
		hiddenNodes.add(biasToOutput);
			
		//Output node layer
		outputNodes=new ArrayList<Node> ();
		for(int i=0;i<outputNodeCount;i++)
		{
			Node node=new Node(4);
			//Connecting output layer nodes with hidden layer nodes
			for(int j=0;j<hiddenNodes.size();j++)
			{
				NodeWeightPair nwp=new NodeWeightPair(hiddenNodes.get(j), outputWeights[i][j]);
				node.parents.add(nwp);
			}	
			outputNodes.add(node);
		}	
	}
	
	/**
	 * Get the output from the neural network for a single instance
	 * Return the idx with highest output values. For example if the outputs
	 * of the outputNodes are [0.1, 0.5, 0.2], it should return 1.
	 * The parameter is a single instance
	 */
	
	public int calculateOutputForInstance(Instance inst)
	{
		// TODO: add code here
		//first we need to deliver the input values from the instance
		//object and put them into the inputNodes array
		for (int i = 0; i < inst.attributes.size(); i++){
			//Note: setInput is only relevant for type 0 nodes aka input nodes
			inputNodes.get(i).setInput(inst.attributes.get(i));
		}
		
		//now go through hidden nodes and calculate their outputs and use
		//those as inputs for the output node and finally calculate output 
		//value at output node. In other words, we are going to get the RELU 
		//value from a given hidden node and 
		for (int i = 0; i < hiddenNodes.size(); i++){
			//Note: remember that calculateOutput internally updates the output of the node. 
			hiddenNodes.get(i).calculateOutput();
		}
		
		
		int returnIndex = 0;//default index... how should we handle if there are no output nodes?
		double currMaxOutput = (double) Integer.MIN_VALUE;//start off at smallest possible number
		
		//go through the output nodes and calculate their outputs one at a time and then determine
		//the index which houses the greatest output
		for (int i = 0; i < outputNodes.size(); i++){
			Node currOutputNode = outputNodes.get(i);
			currOutputNode.calculateOutput();
			if (currMaxOutput < currOutputNode.getOutput()){
				returnIndex = i;
				currMaxOutput = currOutputNode.getOutput();
			}
		}
		
		return returnIndex;
	}
	

	
	
	
	/**
	 * Train the neural networks with the given parameters
	 * 
	 * The parameters are stored as attributes of this class
	 */
	
	public void train()
	{
		// TODO: add code here
	}
}
