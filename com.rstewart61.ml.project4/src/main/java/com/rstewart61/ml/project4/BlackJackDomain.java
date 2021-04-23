package com.rstewart61.ml.project4;

import java.awt.Color;
import java.awt.Graphics2D;
import java.awt.geom.Line2D;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

//Based on https://raw.githubusercontent.com/jmacglashan/burlap_examples/master/src/main/java/edu/brown/cs/burlap/tutorials/domain/simple/ExampleGridWorld.java

import burlap.mdp.auxiliary.DomainGenerator;
import burlap.mdp.core.Domain;
import burlap.mdp.core.StateTransitionProb;
import burlap.mdp.core.TerminalFunction;
import burlap.mdp.core.action.Action;
import burlap.mdp.core.action.UniversalActionType;
import burlap.mdp.core.state.State;
import burlap.mdp.singleagent.SADomain;
import burlap.mdp.singleagent.environment.SimulatedEnvironment;
import burlap.mdp.singleagent.model.FactoredModel;
import burlap.mdp.singleagent.model.RewardFunction;
import burlap.mdp.singleagent.model.statemodel.FullStateModel;
import burlap.shell.visual.VisualExplorer;
import burlap.visualizer.StatePainter;
import burlap.visualizer.StateRenderLayer;
import burlap.visualizer.Visualizer;



public class BlackJackDomain implements DomainGenerator {
	public static int WINDOW_SIZE = 3;
	public static char X = 'X';
	
	protected int size;
	private int numReachableStates = 0;

	public BlackJackDomain(int size) {
		this.size = size;
	}
/*
	@Override
	public SADomain generateDomain() {
		SADomain domain = new SADomain();
		domain.addActionType(new UniversalActionType(BlackJackState.ACTION_HIT));
		domain.addActionType(new UniversalActionType(BlackJackState.ACTION_HOLD));

		BlackJackStateModel smodel = new BlackJackStateModel();
		RewardFunction rf = new BlackJackRF();
		TerminalFunction tf = new BlackJackTF();

		domain.setModel(new FactoredModel(smodel, rf, tf));

		mapState(initialState, 0);
		System.out.println("Num reachable states for size " + size + " is " + numReachableStates);

		return domain;
	}
	
	private void mapState(BlackJackState state, int player) {
		List<Object> keys = state.variableKeys();
		for (Object key : keys) {
			int nextHash = state.getNextHash(key);
			if (cache[nextHash] == null) {
				++numReachableStates;
				BlackJackState nextState = state.getNextState(key);
				cache[nextHash] = nextState; // redundant
				boolean isWinner = nextState.wins();
				if (isWinner) {
					switch (player) {
					case 0:
						nextState.reward = 1;
						break;
					case 1:
						nextState.reward = -100;
						break;
					}
				} else {
					mapState(nextState, (player + 1) % 2);
				}
				nextState.updateKeys();
				// System.out.println("Mapped " + nextState.toString());
			}
		}
	}


	public StateRenderLayer getStateRenderLayer(){
		StateRenderLayer rl = new StateRenderLayer();
		rl.addStatePainter(new BlackJackDomain.AgentPainter());


		return rl;
	}

	public Visualizer getVisualizer(){
		return new Visualizer(this.getStateRenderLayer());
	}


	private int getPosition(Action a) {
		return positionMap.get(a.actionName());
	}

	Random random = new Random();
	protected class BlackJackStateModel implements FullStateModel{
		public BlackJackStateModel() {
		}

		@Override
		public List<StateTransitionProb> stateTransitions(State s, Action a) {
			BlackJackState gs = (BlackJackState)s;
			if (gs.reward != 0) {
				return Arrays.asList(new StateTransitionProb(s, 1.0));
			}

			BlackJackState ns = gs.getNextState(getPosition(a));
			List<Object> keys = ns.variableKeys();
			List<StateTransitionProb> tps = new ArrayList<StateTransitionProb>(keys.size());
			if (ns.reward > 0) {
				// player wins
				tps.add(new StateTransitionProb(ns, 1.0));
			} else {
				for (Object key : keys) {
					BlackJackState nns = ns.getNextState(key);
					tps.add(new StateTransitionProb(nns, 1.0 / keys.size()));
				}
			}

			return tps;
		}

		@Override
		public State sample(State s, Action a) {
			BlackJackState gs = (BlackJackState)s;
			BlackJackState ns = gs.getNextState(getPosition(a));
			if (ns.reward != 0) {
				return ns;
			}
			
			List<Object> keys = ns.variableKeys();
			int pos = random.nextInt(keys.size());
			Object sampledKey = keys.get(pos);
			BlackJackState nns = ns.getNextState(sampledKey);

			return nns;
		}
	}

	public class AgentPainter implements StatePainter {

		@Override
		public void paint(Graphics2D g2, State s,
						  float cWidth, float cHeight) {
			System.out.println(s.toString());
			System.out.println(s.variableKeys());
			g2.setColor(Color.BLACK);
			BlackJackState ts = (BlackJackState) s;
			if (ts.reward > 0) {
				g2.setColor(Color.GREEN);
			} else if (ts.reward < 0) {
				g2.setColor(Color.RED);
			}

			//set up floats for the width and height of our domain
			float fWidth = size;
			float fHeight = 1;

			//determine the width of a single cell on our canvas
			//such that the whole map can be painted
			float width = cWidth / fWidth;
			float height = cHeight / fHeight;

			//top coordinate of cell on our canvas
			//coordinate system adjustment because the java canvas
			//origin is in the top left instead of the bottom right
			float ry = cHeight - height;

			for (int i=0; i<size; ++i) {
				Character value = (Character)s.get(i);
				
				//left coordinate of cell on our canvas
				float rx = i*width;

				if (value.equals(X)) {
					g2.draw(new Line2D.Float(rx, ry, rx+width, ry+height));
					g2.draw(new Line2D.Float(rx+width, ry, rx, ry+height));
				}
			}
		}
	}

	public class BlackJackRF implements RewardFunction {
		@Override
		public double reward(State s, Action a, State sprime) {
			return ((BlackJackState) sprime).reward;
		}
	}

	public class BlackJackTF implements TerminalFunction {
		@Override
		public boolean isTerminal(State s) {
			return ((BlackJackState) s).reward != 0;
		}
	}

	public static void main(String [] args) {
		int boardSize = 10;
		BlackJackDomain gen = new BlackJackDomain(boardSize);
		SADomain domain = gen.generateDomain();
		SimulatedEnvironment env = new SimulatedEnvironment(domain, gen.initialState);

		Visualizer v = gen.getVisualizer();
		VisualExplorer exp = new VisualExplorer(domain, env, v);

		for (int i=0; i<boardSize; ++i) {
			exp.addKeyAction("" + i, "" + i, "");
		}

		exp.initGUI();
	}
	*/

	@Override
	public Domain generateDomain() {
		// TODO Auto-generated method stub
		return null;
	}

}