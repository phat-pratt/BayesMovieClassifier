import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;

/**
 * Your implementation of a naive bayes classifier. Please implement all four methods.
 */

public class NaiveBayesClassifier implements Classifier {
	
	Map<String, Integer> positiveWords = new HashMap<String, Integer>();
	int posSum = 0;
	Map<String, Integer> negativeWords = new HashMap<String, Integer>();
	int negSum = 0;
	Map<Label, Integer> wordCountPerLabel;
	Map<Label, Integer> documentCountPerLabel;
	int gV;
    /**
     * Trains the classifier with the provided training data and vocabulary size
     */
    @Override
    public void train(List<Instance> trainData, int v) {
    	gV = v;
        // TODO : Implement
        // Hint: First, calculate the documents and words counts per label and store them. 
        // Then, for all the words in the documents of each label, count the number of occurrences of each word.
        // Save these information as you will need them to calculate the log probabilities later.
        //
    	getDocumentsCountPerLabel(trainData);
    	getWordsCountPerLabel(trainData);
    	
    	// e.g.
        // Assume m_map is the map that stores the occurrences per word for positive documents
        // m_map.get("catch") should return the number of "catch" es, in the documents labeled positive
        // m_map.get("asdasd") would return null, when the word has not appeared before.
        // Use m_map.put(word,1) to put the first count in.
        // Use m_map.replace(word, count+1) to update the value
    	
    	
    }
    public void clear( ) {
    	positiveWords = new HashMap<String, Integer>();
    	posSum = 0;
    	negativeWords = new HashMap<String, Integer>();
    	negSum = 0;
    }
    public void setSum() {
		negSum = 0;
		for (int f : negativeWords.values()) {
		    negSum += f;
		}
		posSum = 0;
		for (int f : positiveWords.values()) {
		    posSum += f;
		}
    }
    public void getWordCounts(Instance instance) {
    	if(instance.label == Label.POSITIVE) {
    		for(String word : instance.words) {
    			if(positiveWords.containsKey(word)) {
        			positiveWords.put(word, positiveWords.get(word) + 1);
    			} else {
    				positiveWords.put(word, 1);
    			}
    		}
    		
    	} else {
    		for(String word : instance.words) {
    			if(negativeWords.containsKey(word)) {
    				negativeWords.put(word, negativeWords.get(word) + 1);
    			} else {
    				negativeWords.put(word, 1);
    			}
    		}
    	}
    }
    
    /*
     * Counts the number of words for each label
     * 
     *  This method counts the number of words per label in the training set.
     * 
     * Returns - a map that stores the (label, number of words) K-V pair.
     */
    @Override
    public Map<Label, Integer> getWordsCountPerLabel(List<Instance> trainData) {
    	Map<Label, Integer> map = new HashMap<Label, Integer>();
    	map.put(Label.NEGATIVE, 0);
    	map.put(Label.POSITIVE, 0);
    	for(Instance instance : trainData) {
    		map.put(instance.label, map.get(instance.label) + instance.words.size());
    	}
    	wordCountPerLabel = map;
    	setSum();
        return map;
    }

    /*
     * Counts the total number of documents for each label
     * 
     * this method counts the number of reviews per class label in the training set.
     * 
     * Returns - a map that stores the (label, number of documents) K-V pair.
     */
    @Override
    public Map<Label, Integer> getDocumentsCountPerLabel(List<Instance> trainData) {
    	Map<Label, Integer> map = new HashMap<Label, Integer>();
    	map.put(Label.NEGATIVE, 0);
    	map.put(Label.POSITIVE, 0);
    	
    	for(Instance instance : trainData) {
    		map.put(instance.label, map.get(instance.label) + 1);
    		getWordCounts(instance);
    	}
    	documentCountPerLabel = map;
    	return map;
    }


    /**
     * Returns the prior probability of the label parameter, i.e. P(POSITIVE) or P(NEGATIVE)
     */
    private double p_l(Label label) {
        // Calculate the probability for the label. No smoothing here.
        // Just the number of label counts divided by the number of documents.
    	return ((double)documentCountPerLabel.get(label))/(double)(documentCountPerLabel.get(Label.NEGATIVE) 
    															 + documentCountPerLabel.get(Label.POSITIVE));	
    }

    /**
     * Returns the smoothed conditional probability of the word given the label, i.e. P(word|POSITIVE) or
     * P(word|NEGATIVE)
     */
    private double p_w_given_l(String word, Label label) {
    	double pwgl = 0.0;
    	double delta = 1.0;
    	double numerator = 0.0;
    	double denominator = gV;
    	
        // Calculate the probability with Laplace smoothing for word in class(label)
    	
    	if(label == Label.NEGATIVE) {
    		if(negativeWords.containsKey(word)) {
    			//add delta to word count.
    			numerator = negativeWords.get(word) + delta;
    		} else {
    			numerator = delta;
    		}
    		denominator += (double)negSum;
    	} else {
    		if(positiveWords.containsKey(word)) {
    			//add delta to word count.
    			numerator = positiveWords.get(word) + delta;
    		} else {
    			numerator = delta;
    		}
    		denominator += (double)posSum;
    	}
    	pwgl = numerator/denominator;
    	
        return pwgl;
    }

    /**
     * Classifies an array of words as either POSITIVE or NEGATIVE.
     * 
     * Returns the classification result for a single movie review.
     */
    @Override
    public ClassifyResult classify(List<String> words) {
        // Sum up the log probabilities for each word in the input data, and the probability of the label
        // Set the label to the class with larger log probability
    	Map<Label, Double> logProbPerLabel = new HashMap<Label, Double>();
    	logProbPerLabel.put(Label.NEGATIVE, Math.log(p_l(Label.NEGATIVE))); ;
        logProbPerLabel.put(Label.POSITIVE, Math.log(p_l(Label.POSITIVE)));
    	ClassifyResult cr = new ClassifyResult();

        //Add negative log probs
        for (String key : words) {
            logProbPerLabel.put(Label.NEGATIVE, logProbPerLabel.get(Label.NEGATIVE) + Math.log(p_w_given_l(key, Label.NEGATIVE)));
        }
        //Add positive log probs
        for (String key : words) {
            logProbPerLabel.put(Label.POSITIVE, logProbPerLabel.get(Label.POSITIVE) + Math.log(p_w_given_l(key, Label.POSITIVE)));
        }
        //if positive prob is greater than or equal to negative, return POSITIVE
        if(logProbPerLabel.get(Label.POSITIVE) >= logProbPerLabel.get(Label.NEGATIVE)) {
        	cr.label = Label.POSITIVE;
        	cr.logProbPerLabel = logProbPerLabel;
        }
        else{
        	cr.label = Label.NEGATIVE;
        	cr.logProbPerLabel = logProbPerLabel;
        }
        return cr;
    }


}
