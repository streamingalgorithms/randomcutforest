package org.streamingalgorithms.randomcutforest.examples.summarization;

import static java.lang.Math.min;

import java.util.List;
import java.util.Random;

import org.streamingalgorithms.randomcutforest.examples.Example;
import org.streamingalgorithms.randomcutforest.summarization.ICluster;
import org.streamingalgorithms.randomcutforest.summarization.Summarizer;
import org.streamingalgorithms.randomcutforest.util.Weighted;

/**
 * the following example showcases the use of RCF multi-summarization on generic
 * types R, when provided with a distance function from (R,R) into double. In
 * this example R correpsonds to Strings and the distance is EditDistance The
 * srings are genrated from two clusters one where character A (or '-' for viz)
 * occurs with probability 2/3 and anothewr where it occurs with probability 1/3
 * (and the character B or '_' occurs with probability 2/3)
 *
 * Clearly multicentroid approach is necessary.
 *
 */
/*
 * mvn -pl examples -am install -DskipTests java --add-modules
 * jdk.incubator.vector -jar examples/target/*-jar-with-dependencies.jar
 * string_summarize
 */
public class StringSummarization implements Example {

    public static void main(String[] args) throws Exception {
        new StringSummarization().run();
    }

    @Override
    public String command() {
        return "string_summarize";
    }

    @Override
    public String description() {
        return "Example of string summarization using multi-centroid approach";
    }

    @Override
    public void run() throws Exception {

        long seed = new Random().nextLong();
        System.out.println("String summarization seed : " + seed);
        Random random = new Random(seed);
        int size = 100;
        int numberOfStrings = 20000;

        String[] points = new String[numberOfStrings];
        for (int i = 0; i < numberOfStrings; i++) {
            if (random.nextDouble() < 0.5) {
                points[i] = getABString(size, 0.8, random);
            } else {
                points[i] = getABString(size, 0.2, random);
            }
        }

        int nextSeed = random.nextInt();
        List<ICluster<String>> summary = Summarizer.multiSummarize(points, 5, 10, 1, false, 0.8,
                StringSummarization::toyDistance, nextSeed, true, 0.1, 5);
        System.out.println();
        for (int i = 0; i < summary.size(); i++) {
            double weight = summary.get(i).getWeight();
            System.out.println(
                    "Cluster " + i + " representatives, weight " + ((float) Math.round(1000 * weight) * 0.001));
            List<Weighted<String>> representatives = summary.get(i).getRepresentatives();
            for (int j = 0; j < representatives.size(); j++) {
                double t = representatives.get(j).weight;
                t = Math.round(1000.0 * t / weight) * 0.001;
                System.out.print(
                        "relative weight " + (float) t + " length " + representatives.get(j).index.length() + " ");
                printString(representatives.get(j).index);
                System.out.println();
            }
            System.out.println();
        }

    }

    public static double toyDistance(String a, String b) {
        if (a.length() > b.length()) {
            return toyDistance(b, a);
        }
        double[][] dist = new double[2][b.length() + 1];
        for (int j = 0; j < b.length() + 1; j++) {
            dist[0][j] = j;
        }

        for (int i = 1; i < a.length() + 1; i++) {
            dist[1][0] = i;
            for (int j = 1; j < b.length() + 1; j++) {
                double t = dist[0][j - 1] + ((a.charAt(i - 1) == b.charAt(j - 1)) ? 0 : 1);
                dist[1][j] = min(min(t, dist[0][j] + 1), dist[1][j - 1] + 1);
            }
            for (int j = 0; j < b.length() + 1; j++) {
                dist[0][j] = dist[1][j];
            }
        }
        return dist[1][b.length()];
    }

    public static final String ANSI_RESET = "\u001B[0m";
    public static final String BG_RED = "\u001B[41m"; // red background
    public static final String BG_BLUE = "\u001B[44m"; // blue background

    public static void printString(String a) {
        StringBuilder sb = new StringBuilder();
        char prev = 0;
        for (int i = 0; i < a.length(); i++) {
            char c = a.charAt(i);
            String bg = (c == '-') ? BG_RED : BG_BLUE;
            if (c != prev) { // only emit a code when the run flips
                sb.append(bg);
                prev = c;
            }
            sb.append(' '); // space fills the whole cell -> continuous
        }
        sb.append(ANSI_RESET);
        System.out.print(sb);
    }

    public String getABString(int size, double probabilityOfA, Random random) {
        StringBuilder stringBuilder = new StringBuilder();
        int newSize = size + random.nextInt(size / 5);
        for (int i = 0; i < newSize; i++) {
            if (random.nextDouble() < probabilityOfA) {
                stringBuilder.append("-");
            } else {
                stringBuilder.append("_");
            }
        }
        return stringBuilder.toString();
    }
}
