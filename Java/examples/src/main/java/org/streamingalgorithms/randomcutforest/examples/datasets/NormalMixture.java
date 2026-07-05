package org.streamingalgorithms.randomcutforest.examples.datasets;

import java.util.Arrays;
import java.util.Random;

public class NormalMixture {

    private final double baseMu;
    private final double baseSigma;
    private final double anomalyMu;
    private final double anomalySigma;
    private final double transitionToAnomalyProbability;
    private final double transitionToBaseProbability;

    public NormalMixture(double baseMu, double baseSigma, double anomalyMu, double anomalySigma,
            double transitionToAnomalyProbability, double transitionToBaseProbability) {
        this.baseMu = baseMu;
        this.baseSigma = baseSigma;
        this.anomalyMu = anomalyMu;
        this.anomalySigma = anomalySigma;
        this.transitionToAnomalyProbability = transitionToAnomalyProbability;
        this.transitionToBaseProbability = transitionToBaseProbability;
    }

    public NormalMixture() {
        this(0.0, 1.0, 4.0, 2.0, 0.01, 0.3);
    }

    public NormalMixture(double baseMu, double anomalyMu) {
        this(baseMu, 1.0, anomalyMu, 2.0, 0.01, 0.3);
    }

    public MultiDimData generateData(int numberOfRows, int numberOfColumns, int seed) {
        float[][] result = new float[numberOfRows][numberOfColumns];
        int[] change = new int[numberOfRows];
        int numberOfChanges = 0;
        boolean anomaly = false;

        Random gen = (seed != 0) ? new Random(seed) : new Random();
        Random anomalyGen = new Random(gen.nextLong());
        NormalDistribution dist = new NormalDistribution(gen);
        for (int i = 0; i < numberOfRows; i++) {
            if (!anomaly) {
                fillRow(result[i], dist, baseMu, baseSigma);
                if (anomalyGen.nextDouble() < transitionToAnomalyProbability) {
                    change[numberOfChanges++] = i + 1;
                    anomaly = true;
                }
            } else {
                fillRow(result[i], dist, anomalyMu, anomalySigma);
                if (anomalyGen.nextDouble() < transitionToBaseProbability) {
                    anomaly = false;
                    change[numberOfChanges++] = i + 1;
                }
            }
        }

        return new MultiDimData(result, Arrays.copyOf(change, numberOfChanges), null);
    }

    private void fillRow(float[] row, NormalDistribution dist, double mu, double sigma) {
        for (int j = 0; j < row.length; j++) {
            row[j] = (float) (dist.nextDouble(mu, sigma));
        }
    }

    public static class NormalDistribution {
        private final Random rng;
        private final double[] buffer;
        private int index;

        public NormalDistribution(Random rng) {
            this.rng = rng;
            buffer = new double[2];
            index = 0;
        }

        public double nextDouble() {
            if (index == 0) {
                // apply the Box-Muller transform to produce Normal variates
                double u = rng.nextDouble();
                double v = rng.nextDouble();
                double r = Math.sqrt(-2 * Math.log(u));
                buffer[0] = r * Math.cos(2 * Math.PI * v);
                buffer[1] = r * Math.sin(2 * Math.PI * v);
            }

            double result = buffer[index];
            index = (index + 1) % 2;

            return result;
        }

        public double nextDouble(double mu, double sigma) {
            return mu + sigma * nextDouble();
        }
    }
}
