package com.rstewart61.ml.project4;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.LinkedList;
import java.util.List;
import java.util.Queue;

// Based on https://raw.githubusercontent.com/jmacglashan/burlap_examples/master/src/main/java/edu/brown/cs/burlap/tutorials/domain/simple/EXGridState.java

import burlap.mdp.core.state.MutableState;
import burlap.mdp.core.state.UnknownKeyException;
import burlap.mdp.core.state.annotations.DeepCopyState;


@DeepCopyState
public class BlackJackState implements MutableState {
	public final static String ACTION_HIT = "HIT";
	public final static String ACTION_HOLD = "HOLD";
	public final static String ACTION_DEAL = "DEAL";
	public static List<Object> actions = Arrays.asList(ACTION_DEAL, ACTION_HIT, ACTION_HOLD);
	
	public final static String VAR_PLAYER_UP = "PLAYER_UP";
	public final static String VAR_PLAYER_DOWN = "PLAYER_DOWN";
	public final static String VAR_PLAYER_HIT_LIST = "HIT_LIST";
	public final static String VAR_DEALER_UP = "DEALER_UP";
	public final static String VAR_OTHER_CARDS = "OTHER_CARDS";
	private static List<Object> variables = Arrays.asList(VAR_PLAYER_UP, VAR_PLAYER_DOWN,
			VAR_PLAYER_HIT_LIST, VAR_DEALER_UP, VAR_OTHER_CARDS);

	
	public static List<Integer> CARDS = Arrays.asList(2, 3, 4, 5, 6, 7, 8, 9,
			10, /*J=*/10, /*Q=*/10, /*K=*/10, /*A=*/11);
	
	private int upCard;
	private int downCard;
	private int dealerCard;
	private List<Integer> hitList = new ArrayList<>();
	private List<Integer> otherList = new ArrayList<>();
	private Queue<Integer> availableCards = new LinkedList<>(CARDS);
	private int playerTotal = 0;
	private int dealerTotal = 0;
	public int reward = 0;
	
	int size;

	public BlackJackState(int size) {
		this.size = size;
	}
	
	private void deal() {
		List<Integer> cardShuffle = new ArrayList<>(CARDS);
		Collections.shuffle(cardShuffle);
		availableCards = new LinkedList<>(cardShuffle);

		upCard = availableCards.remove();
		downCard = availableCards.remove();
		playerTotal = upCard + downCard;
		dealerCard = availableCards.remove();
		dealerTotal = dealerCard;
		for (int i=0; i<size-3; ++i) {
			otherList.add(availableCards.remove());
		}
	}

	public BlackJackState(BlackJackState other) {
		this.upCard = other.upCard;
		this.downCard = other.downCard;
		this.playerTotal = other.playerTotal;
		this.dealerCard = other.dealerCard;
		this.dealerTotal = other.dealerTotal;
		this.hitList = new ArrayList<>(other.hitList);
		this.otherList = other.otherList;
		this.availableCards = new LinkedList<>(other.availableCards);
	}
	
	@Override
	public BlackJackState set(Object variableKey, Object value) {
		if (variableKey.equals(ACTION_DEAL)) {
			deal();
		} else if (variableKey.equals(ACTION_HIT)) {
			int hitCard = availableCards.remove();
			this.hitList.add(hitCard);
			this.playerTotal += hitCard;
			if (playerTotal > 21) {
				playerTotal = -21;
			}
		} else if (variableKey.equals(ACTION_HOLD)) {
			int dealerDown = 0;
			do {
				dealerDown = availableCards.remove();
				dealerTotal += dealerDown;
			} while (dealerTotal < 17);
			if (dealerTotal > 21) {
				dealerTotal = 0;
			}
			this.reward = playerTotal - dealerTotal;
		}
		return this;
	}
	
	public List<Object> variableKeys() {
		return variables;
	}

	@Override
	public Object get(Object variableKey) {
		switch (variableKey.toString()) {
		case VAR_PLAYER_UP:
			return upCard;
		case VAR_PLAYER_DOWN:
			return downCard;
		case VAR_PLAYER_HIT_LIST:
			return hitList;
		case VAR_DEALER_UP:
			return dealerCard;
		case VAR_OTHER_CARDS:
			return otherList;
		}
		throw new UnknownKeyException(variableKey);
	}

	@Override
	public BlackJackState copy() {
		return new BlackJackState(this);
	}

	@Override
	public String toString() {
		return "BlackJackState [upCard=" + upCard + ", downCard=" + downCard + ", dealerCard=" + dealerCard
				+ ", hitList=" + hitList + ", otherList=" + otherList + ", availableCards=" + availableCards
				+ ", playerTotal=" + playerTotal + ", dealerTotal=" + dealerTotal + ", size=" + size + "]";
	}
	
	public int hashCode() {
		throw new UnsupportedOperationException("");
	}
	
	public boolean equals(Object other) {
		throw new UnsupportedOperationException("");
	}
}

