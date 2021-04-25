# Machine Learning Assignment 4

# First, get the package:
git clone https://github.com/rstewart61/gt-mdp.git
cd gt-mdp

# Install basic requirements:
pip install -r requirements.txt

# Make sure to have maven and a recent JDK installed.

# To generate raw data, run:
cd com.rstewart61.ml.project4
mvn package assembly:single
java -Xmx18g -jar target/com.rstewart61.ml.project4-0.0.1-SNAPSHOT-jar-with-dependencies.jar ../output/

# Expect this to run for several hours, depending on your CPU.

# To generate plots used for the analysis report, run:
python3 plot_results.py

# If you have ImageMagick, plots can be glued together with:
./glue_plots.sh

