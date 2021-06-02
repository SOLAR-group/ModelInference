

export KLFA_PATH=$KLFA_HOME/klfa-standalone.jar
export SLCT_BIN=/opt/slct/slct

java -cp $KLFA_PATH it.unimib.disco.lta.alfa.klfa.LogTraceAnalyzer -separator "," -outputDir klfaCheckingApplicationLevel -inputDir klfaTrainingApplication/kate applicationLevel checking transformersConfig.txt preprocessingRules.txt datasets/test/kate_plus.csv

# -minimizationLimit 100 
