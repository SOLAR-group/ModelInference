package regressionTestManager.ioInvariantParser;

import sjm.parse.Assembler;
import sjm.parse.Assembly;

public class DivisionDoubleExpressionAssembler extends Assembler{

  public void workOn(Assembly a) {
    Target target = (Target)a.getTarget();
    //Double z = (Double)target.pop();
    //Double y = (Double)target.pop();
    //Double x = (Double)target.pop();
    Double z = null;
    try{
    	z = (Double)target.pop();
    }catch(ArrayIndexOutOfBoundsException e) {
      	EvaluationRuntimeErrors.emptyStack();
      	return;
    }
    Double y = null;
    try{
    	y = (Double)target.pop();
    }catch(ArrayIndexOutOfBoundsException e) {
      	EvaluationRuntimeErrors.emptyStack();
      	return;
    }
    Double x = null;
    try{
    	x = (Double)target.pop();
    }catch(ArrayIndexOutOfBoundsException e) {
      	EvaluationRuntimeErrors.emptyStack();
      	return;
    }
    try{
    	EvaluationRuntimeErrors.log("Division double not implemented");
    	
    //target.push(new Boolean(y.longValue() / z.longValue() == x.longValue()));
    }
    catch (Exception e) {
		EvaluationRuntimeErrors.evaluationError();
		//target.push(Boolean.FALSE);
		return;
    }
  }
}