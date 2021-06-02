export KLFA_PATH=$KLFA_HOME/klfa-standalone.jar
export SLCT_BIN=/opt/slct/slct

java -Xmx8g -cp $KLFA_PATH it.unimib.disco.lta.alfa.klfa.LogTraceAnalyzer -separator ","  -outputDir klfaTrainingApplication/kate applicationLevel training transformersConfig.txt preprocessingRules.txt datasets/train/kate_plus.csv

#-minimizationLimit 100 -fsaFileFormat
