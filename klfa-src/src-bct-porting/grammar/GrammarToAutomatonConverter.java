/* -- JFLAP 4.0 --
 *
 * Copyright information:
 *
 * Susan H. Rodger, Thomas Finley
 * Computer Science Department
 * Duke University
 * April 24, 2003
 * Supported by National Science Foundation DUE-9752583.
 *
 * Copyright (c) 2003
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms are permitted
 * provided that the above copyright notice and this paragraph are
 * duplicated in all such forms and that any documentation,
 * advertising materials, and other materials related to such
 * distribution and use acknowledge that the software was developed
 * by the author.  The name of the author may not be used to
 * endorse or promote products derived from this software without
 * specific prior written permission.
 * THIS SOFTWARE IS PROVIDED ``AS IS'' AND WITHOUT ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, WITHOUT LIMITATION, THE IMPLIED
 * WARRANTIES OF MERCHANTIBILITY AND FITNESS FOR A PARTICULAR PURPOSE.
 */
 
package grammar;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;

import automata.Automaton;
import automata.State;
import automata.Transition;

/**
 * The grammar to automaton converter can be used to convert a grammar
 * to an equivalent automaton.  This is an abstract class that is 
 * extended by converters for specific types of grammars (e.g. regular)
 * To utilize this converter, you must determine what type of grammar
 * you are trying to convert by using a GrammarChecker object and then
 * instantiate the appropriate converter (e.g. RegularGrammarToFSAConverter).
 * Then you must instantiate an automaton (of the type that the converter
 * will create), and then call convertToAutomaton.  Or you can do it 
 * yourself by calling the createStatesForConversion method.  
 * Then you can call the getTransitionsForProduction method for each 
 * production in the grammar and add them to your automaton.
 *
 * @author Ryan Cavalcante
 */

public abstract class GrammarToAutomatonConverter {
    /**
     * Instantiates a <CODE>GrammarToAutomatonConverter</CODE>.
     */
    public GrammarToAutomatonConverter() {
	
    }

    /**
     * Initializes the converter for a new conversion by 
     * clearing its map.
     */
    public void initialize() {
	MAP = new HashMap();
    }

    /**
     * Returns the State object mapped to <CODE>variable</CODE>
     * @param variable the variable
     * @return the State object mapped to <CODE>variable</CODE>
     */
    public State getStateForVariable(String variable) {
	return (State) MAP.get(variable);
    }

    /**
     * Maps <CODE>state</CODE> to <CODE>variable</CODE>
     * @param state the state
     * @param variable the variable
     */
    public void mapStateToVariable(State state, String variable) {
	MAP.put(variable, state);
    }

    /**
     * Returns the transition created by converting 
     * <CODE>production</CODE> to its equivalent transition.
     * @param production the production
     * @return the equivalent transition.
     */
    public abstract Transition getTransitionForProduction
	(Production production);

    /**
     * Returns an automaton that is equivalent to <CODE>grammar</CODE>
     * in that they will accept the same language.
     * @param grammar the grammar
     * @return an automaton that is equivalent to <CODE>grammar</CODE>
     * in that they will accept the same language.
     */
    public Automaton convertToAutomaton(Grammar grammar) {
	ArrayList list = new ArrayList();
	Automaton automaton = new Automaton();
	createStatesForConversion(grammar, automaton);
	Production[] productions = grammar.getProductions();
	for(int k = 0; k < productions.length; k++) {
	    list.add(getTransitionForProduction(productions[k]));
	}
	
	Iterator it = list.iterator();
	while(it.hasNext()) {
	    Transition transition = (Transition) it.next();
	    automaton.addTransition(transition);
	}
	return automaton;
    }
    
    /**
     * Adds all states to <CODE>automaton</CODE> necessary for the 
     * conversion of <CODE>grammar</CODE> to its equivalent
     * automaton.  This creates a state for each variable in 
     * <CODE>grammar</CODE> and maps each created state to the
     * variable it was created for by calling mapStateToVariable.
     * @param grammar the grammar being converted.
     * @param automaton the automaton being created.
     */
    public abstract void createStatesForConversion(Grammar grammar, 
						   Automaton automaton);

    protected HashMap MAP;
    protected String BOTTOM_OF_STACK = "Z";
}
