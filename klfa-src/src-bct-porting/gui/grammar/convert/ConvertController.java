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
 
package gui.grammar.convert;

import grammar.Grammar;
import grammar.Production;
import gui.event.SelectionEvent;
import gui.event.SelectionListener;
import gui.viewer.SelectionDrawer;

import java.awt.Component;
import java.util.Collection;
import java.util.HashSet;
import java.util.Iterator;
import java.util.Map;
import java.util.Set;

import automata.Automaton;
import automata.Transition;
import automata.event.AutomataTransitionEvent;
import automata.event.AutomataTransitionListener;

/**
 * A <CODE>ConvertController</CODE> object is a controller used in the
 * conversion of a grammar to some sort of automaton.  It monitors
 * both the grammar and the automaton being built, as well as their
 * respective views.  It intervenes as necessary, and as it is a
 * controller object moderates between the views, and the automaton
 * and grammar.
 * 
 * @see grammar.Grammar
 * @see automaton.Automaton
 * @see gui.grammar.GrammarView
 * @see gui.viewer.SelectionDrawer
 * 
 * @author Thomas Finley
 */

class ConvertController {
    /**
     * Instantiates a <CODE>ConvertController</CODE> object.
     * @param grammarView the grammar view
     * @param drawer the automaton selection drawer
     * @param productionsToTransitions a map from productions to the
     * corresponding transitions the user should come up with... this
     * maping must be one-to-one
     * @param parent some parent object so that the controller knows
     * where to put its message boxes, which may be null
     */
    public ConvertController(GrammarViewer grammarView,
			     SelectionDrawer drawer,
			     Map productionsToTransitions,
			     Component parent) {
	this.grammarView = grammarView;
	this.drawer = drawer;
	this.parent = parent;
	grammar = grammarView.getGrammar();
	automaton = drawer.getAutomaton();

	initListeners();
	pToT = productionsToTransitions;
	tToP = invert(pToT);
    }

    /**
     * Returns a map containing the inverse of the passed in map.
     * @param map the map, which should be one to one
     * @return the inverse of the passed in map, or <CODE>null</CODE>
     * if an error occurred
     */
    private Map invert(Map map) {
	Set entries = map.entrySet();
	Iterator it = entries.iterator();
	Map inverse = null;
	try {
	    inverse = (Map) map.getClass().newInstance();
	    while (it.hasNext()) {
		Map.Entry entry = (Map.Entry) it.next();
		inverse.put(entry.getValue(), entry.getKey());
	    }
	} catch (Throwable e) {
	    
	}
	return inverse;
    }

    /**
     * This initializes the listeners to the GUI objects, as well as
     * the automata.
     */
    private void initListeners() {
	automaton.addTransitionListener(new AutomataTransitionListener() {
		public void automataTransitionChange
		    (AutomataTransitionEvent e) {
		    if (!e.isAdd()) {
			return;
		    }
		    Transition transition = e.getTransition();
		    if (!tToP.containsKey(transition) ||
			alreadyDone.contains(tToP.get(transition))) {
			javax.swing.JOptionPane.showMessageDialog
			    (parent, "That transition is not correct!");
			automaton.removeTransition(transition);
		    } else {
			Production p = (Production) tToP.get(transition);
			alreadyDone.add(p);
			grammarView.setChecked(p, true);
		    }
		}
	    });

	grammarView.addSelectionListener(new SelectionListener() {
		public void selectionChanged(SelectionEvent event) {
		    Production[] p = grammarView.getSelected();
		    drawer.clearSelected();
		    for (int i=0; i<p.length; i++) {
			drawer.addSelected((Transition) pToT.get(p[i]));
		    }
		    parent.repaint();
		}
	    });
    }

    /**
     * Puts all of the remaining uncreated transitions into the
     * automaton.
     */
    public void complete() {
	Collection productions = new HashSet(pToT.keySet());
	Iterator it = productions.iterator();
	while (it.hasNext()) {
	    Production p = (Production) it.next();
	    if (alreadyDone.contains(p)) continue;
	    Transition t = (Transition) pToT.get(p);
	    automaton.addTransition(t);
	}
    }

    /**
     * Puts all of the transitions for the selected productions in the
     * automaton.
     */
    public void createForSelected() {
	Production[] p = grammarView.getSelected();
	for (int i=0; i<p.length; i++) {
	    if (alreadyDone.contains(p[i])) continue;
	    Transition t = (Transition) pToT.get(p[i]);
	    automaton.addTransition(t);
	}
    }

    /**
     * Displays and returns if the automaton is done yet.
     * @return <CODE>true</CODE> if the automaton is done,
     * <CODE>false</CODE> if it is not
     */
    public boolean isDone() {
	int toDo = pToT.size() - alreadyDone.size();
	String message = toDo == 0 ?
	    "The conversion is finished!" :
	    toDo+" more transition"+(toDo==1 ? "" : "s")+" must be added.";
	javax.swing.JOptionPane.showMessageDialog
	    (parent, message);
	
	return toDo == 0;
    }

    /**
     * If the conversion is done, this takes the automaton and makes
     * it editable in a new window.
     */
    public void export() {
	boolean done = (pToT.size() - alreadyDone.size())==0;
	if (!done) {
	    javax.swing.JOptionPane.showMessageDialog
		(parent,"The conversion is not completed yet!");
	    return;
	}
	Automaton toExport = (Automaton) automaton.clone();
	gui.environment.FrameFactory.createFrame(toExport);
    }

    /** The grammar view. */
    protected GrammarViewer grammarView;
    /** The automaton drawer. */
    protected SelectionDrawer drawer;
    /** The grammar. */
    protected Grammar grammar;
    /** The automaton. */
    protected Automaton automaton;
    /** The map of productions to transitions the user should come up with. */
    protected Map pToT;
    /** The map of transitions to productions. */
    protected Map tToP;

    /** The set of productions whose transitions have already been
     * added. */
    protected Set alreadyDone = new HashSet();

    /** The parent component. */
    protected Component parent;
}
