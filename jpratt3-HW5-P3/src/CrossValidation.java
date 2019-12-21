import java.util.ArrayList;
import java.util.List;

public class CrossValidation {
    /*
     *This method take a classifier and preforms k-fold cross validation on traindata
     *The output is the average of the scores computed on each fold. 
     *
     *Assume that the number of instances in traindata is always divisible by k. 
     *(so the size of each fold will be the same.) 
     * 
     * Returns the k-fold cross validation score of classifier clf on training data.
     */
    public static double kFoldScore(Classifier clf, List<Instance> trainData, int k, int v) {
    	
    	int partitionSize = trainData.size()/k;
    	double foldAverage = 0.0;
    	for(int i = 0; i < k; i++) {
        	List<Instance> trainSet = new ArrayList<Instance>();
        	List<Instance> foldSet = new ArrayList<Instance>();

        	// take the sublist of fold i
        	for(int j = 0; j<trainData.size(); j++) {
        		if(j >= i*partitionSize && j <(i+1)*partitionSize) {
            		foldSet.add(trainData.get(j));
        		}
        		else {
        			trainSet.add(trainData.get(j));
        		}
        	}
        	clf.train(trainSet, v);
        	double foldCorrect = 0.0;
        	for(Instance instance : foldSet) {
        		ClassifyResult cr = clf.classify(instance.words);
                if(cr.label == instance.label) {
                	foldCorrect++;
                }
        	}
        	foldAverage += foldCorrect/foldSet.size();
        	((NaiveBayesClassifier)clf).clear();
    	}
    	return foldAverage/k;
    }
}
