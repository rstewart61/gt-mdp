package com.rstewart61.ml.project4;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Deque;
import java.util.List;

// Based on https://raw.githubusercontent.com/jmacglashan/burlap_examples/master/src/main/java/edu/brown/cs/burlap/tutorials/domain/simple/EXGridState.java

import burlap.mdp.core.state.MutableState;
import burlap.mdp.core.state.UnknownKeyException;
import burlap.mdp.core.state.annotations.DeepCopyState;


@DeepCopyState
public class TrebleCrossState implements MutableState {
	public final static String VAR_STATE = "STATE";
	/*
	public final static TrebleCrossState WIN = new TrebleCrossState(0, null);
	public final static TrebleCrossState LOSE = new TrebleCrossState(0, null);
	*/
	StringBuilder data = null;
	List<Object> keys = null;
	TrebleCrossState[] cache;
	int size;
	int hash = 0;
	int reward = 0; // +1 for win, -1 for loss.

	public TrebleCrossState(int size, TrebleCrossState[] cache) {
		data = new StringBuilder(size);
		for (int i=0; i<size; ++i) {
			data.append('.');
		}
		// System.out.println(size + " data: '" + data.toString() + "'");
		this.cache = cache;
		this.size = size;
	}

	public TrebleCrossState(TrebleCrossState other) {
		this.data = other.data;
		this.cache = other.cache;
		this.size = other.size;
		this.hash = other.hash;
		this.reward = other.reward;
	}
	
	public int getNextHash(Object key) {
		Integer i = (Integer) key;
		int newHash = hash | (1 << i);
		return newHash;
	}
	
	public TrebleCrossState getNextState(Object key) {
		Integer i = (Integer) key;
		int newHash = hash | (1 << i);
		if (cache[newHash] == null) {
			cache[newHash] = this.copy();
			cache[newHash].data = new StringBuilder(data);
			cache[newHash].data.setCharAt(i, TrebleCrossDomain.X);
			cache[newHash].hash = newHash;
			cache[newHash].updateKeys();
			cache[newHash].reward = reward;
			// System.out.println("HASH: " + toString() + " -" + key + "-> " + cache[newHash].toString());
		}
		/*
		if (cache[hash] != null && cache[hash].data.charAt(i) == TrebleCrossDomain.X) {
			throw new IllegalArgumentException("Cannot set " + key + " for " + toString());
		}
		*/
		return cache[newHash];
	}

	@Override
	public TrebleCrossState set(Object variableKey, Object value) {
		// Value is ignored, can only set X's.
		throw new RuntimeException("don't do this");
		
		/*
		if (variableKey instanceof Integer) {
			Integer i = (Integer) variableKey;
			int newHash = hash & (1 << i);
			if (cache[newHash] != null) {
				data = cache[newHash].data;
				keys = cache[newHash].keys;
				hash = cache[newHash].hash;
			} else {
				data = new StringBuilder(data);
				updateKeys();
				hash = newHash;
				cache[newHash] = this;
			}
			data.setCharAt(i, TrebleCrossDomain.X);
		}
		else{
			throw new UnknownKeyException(variableKey);
		}
		return this;
		*/
	}

	private static List<Object> emptyKeys = new ArrayList<>();
	
	public void updateKeys() {
		if (keys != null) return;
		if (reward != 0) {
			keys = emptyKeys;
			return;
		}
		keys = new ArrayList<>();
		for (int i=0; i<data.length(); ++i) {
			if (data.charAt(i) != TrebleCrossDomain.X) {
				keys.add(Integer.valueOf(i));
			}
		}
		// System.out.println("Keys for " + toString() + " are " + keys);
	}
	
	public List<Object> variableKeys() {
		updateKeys();
		// System.out.println("Keys are " + keys);
		return keys;
	}

	@Override
	public Object get(Object variableKey) {
		if (variableKey instanceof Integer) {
			Integer i = (Integer) variableKey;
			return data.charAt(i);
		}
		throw new UnknownKeyException(variableKey);
	}

	@Override
	public TrebleCrossState copy() {
		return new TrebleCrossState(this);
	}

	@Override
	public String toString() {
		String extra = " NOT DONE";
		if (reward > 0) {
			extra = " WINS";
		}
		if (reward < 0) {
			extra = " LOSES";
		}
		return data.toString() + extra + " [" + hash + "]";
		// return StateUtilities.stateToString(this);
	}
	
	public int hashCode() {
		return hash;
		/*
		int result = 0;
		int bit = 1;
		for (int i=0; i<size; ++i) {
			result |= bit;
			bit <<= 1;
		}
		return result;
		*/
	}
	
	public boolean equals(Object other) {
		return hash == other.hashCode();
	}
	
	public boolean loses() {
		int maxInRow = 0;
		int currSum = 0;
		Deque<Integer> window = new ArrayDeque<>();
		for (int i=0; i<data.length(); ++i) {
			int currVal = 0;
			if (data.charAt(i) == TrebleCrossDomain.X) {
				currVal = 1;
			}
			window.addFirst(currVal);
			currSum += currVal;
			if (window.size() > TrebleCrossDomain.WINDOW_SIZE) {
				currSum -= window.removeLast();
			}
			maxInRow = Math.max(currSum, maxInRow);
		}
		return maxInRow == TrebleCrossDomain.WINDOW_SIZE;
	}
}

