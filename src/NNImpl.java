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
		
		//System.out.println("This guy: ");
		//go through the output nodes and calculate their outputs one at a time and then determine
		//the index which houses the greatest output
		for (int i = 0; i < outputNodes.size(); i++){
			Node currOutputNode = outputNodes.get(i);
			currOutputNode.calculateOutput();
			//System.out.println("\t" + i + " " + currOutputNode.getOutput());
			if (currMaxOutput <= currOutputNode.getOutput()){
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
		// TODO: add code here DIRECTLY FROM PPT slides
		
		//TODO: check assumption that we only randomize once as opposed to on every epoch
		//like the book says
		
		//go through the number of specified epoch's
		for (int e = 0; e < this.maxEpoch; e++){
			
			for (Instance inst: trainingSet){
			
				//forward pass
				double result = calculateOutputForInstance(inst);//O value in powerpoint slide 43
				//double teacherVal = getTeacherValue(inst);//T value from pp
				//double errorVal = teacherVal - result;

				//backward pass
				double hiddenSumTerm[] = new double[hiddenNodes.size()];
				double deltasFromJK[][] = new double[hiddenNodes.size()][outputNodes.size()];
				for (int k = 0; k < outputNodes.size(); k++){
					Node currOutNode = outputNodes.get(k);
					double errorVal = inst.classValues.get(k) - currOutNode.getOutput();
					double gPrimeJK = ((currOutNode.getOutput()) > 0) ? 1.0: 0.0;
					for (int j = 0; j < currOutNode.parents.size(); j++){
						NodeWeightPair hiddenNode = currOutNode.parents.get(j);
						//delta = alpha * a_j * error * g'(in_k)
						double deltaWeightHtoO = this.learningRate * hiddenNode.node.getOutput()*errorVal*gPrimeJK;
						//summation over k of w_j,k * error * g'(in_k)
						if (k == 0) hiddenSumTerm[j] = 0;
						hiddenSumTerm[j] += hiddenNode.weight * errorVal * gPrimeJK;
						deltasFromJK[j][k] = deltaWeightHtoO;
					}
				}
				
				//forward pass
				double deltasFromIJ[][] = new double[inputNodes.size()][hiddenNodes.size()];
				for (int j = 0; j < (hiddenNodes.size() - 1); j++){
					Node currHidNode = hiddenNodes.get(j);
					double gPrimeJ = ((currHidNode.getOutput()) > 0) ? 1.0: 0.0;
					for (int i = 0; i < currHidNode.parents.size(); i++){
						NodeWeightPair inputNode = currHidNode.parents.get(i);
						//delta = alpha * a_i * g'(in_j) * summation
						double deltaWeightItoH = this.learningRate * inputNode.node.getOutput() * gPrimeJ * hiddenSumTerm[j];
						deltasFromIJ[i][j] = deltaWeightItoH;
					}
					
				}
				
				//apply deltas
				for (int j = 0; j < outputNodes.size(); j++){
					Node currOutNode = outputNodes.get(j);
					
					for (int k = 0; k < currOutNode.parents.size(); k++){
						NodeWeightPair hiddenNode = currOutNode.parents.get(k);
						hiddenNode.weight += deltasFromJK[k][j];
					}
				}
				//apply other deltas
				for (int j = 0; j < (hiddenNodes.size() - 1); j++){
					Node currHidNode = hiddenNodes.get(j);
					for (int i = 0; i < currHidNode.parents.size(); i++){
						NodeWeightPair inputNode = currHidNode.parents.get(i);
						inputNode.weight += deltasFromIJ[i][j];
					}
					
				}
			}
			//code that divides the weights byt he size of the training set
			/*
			for (int j = 0; j < outputNodes.size(); j++){
				Node currOutNode = outputNodes.get(j);
				
				for (int k = 0; k < currOutNode.parents.size(); k++){
					NodeWeightPair hiddenNode = currOutNode.parents.get(k);
					hiddenNode.weight = hiddenNode.weight/trainingSet.size();
				}
			}
			//apply other deltas
			for (int j = 0; j < (hiddenNodes.size() - 1); j++){
				Node currHidNode = hiddenNodes.get(j);
				for (int i = 0; i < currHidNode.parents.size(); i++){
					NodeWeightPair inputNode = currHidNode.parents.get(i);
					inputNode.weight = inputNode.weight/trainingSet.size();
				}
			}
			*/
		}
	}
	
	/**
	 * loops through all the class values of an instance looking for the index where
	 * the value is set to 1. i.e. if [1 0 0] is the contents of the list, then
	 * we return 0 as the index
	*/
	private int getTeacherValue(Instance inst){
		for (int i = 0; i < inst.classValues.size(); i++){
			if (inst.classValues.get(i) == 1){
				return i;
			}
		}
		//error code. There must be a value that is == 1 i.e. there must be a class
		//assigned to the given instance
		return -1;
	}
}
