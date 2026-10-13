// Generates the CDK reference used by FerrumChem's CDK parity tests.
//
// Reads crates/ferrum_core/tests/data/parity_molecules.smi and writes
// crates/ferrum_core/tests/data/cdk_parity_reference.csv with one row per molecule.
// Columns use FerrumChem's feature names where FerrumChem has (or will get) the
// feature; descriptors whose output layout differs from FerrumChem's are emitted
// under CDK's own descriptor names. Fingerprints are stored as
// space-separated on-bit indices (0-based).
//
// Molecules are read, processed and written one at a time (constant memory).
//
//     scripts/cdk_reference/run.sh            # (re)write the reference
//
// Molecule preparation (identical for every descriptor):
//   1. SmilesParser(SilentChemObjectBuilder) - kekulises aromatic SMILES and
//      marks ring atoms/bonds (Cycles.markRingAtomsAndBonds).
//   2. AtomContainerManipulator.percieveAtomTypesAndConfigureAtoms
//   3. CDKHydrogenAdder.addImplicitHydrogens (hydrogens stay implicit)
//   4. new Aromaticity(ElectronDonation.daylight(), Cycles.or(Cycles.all(), Cycles.all(6))).apply
//      This is the model recommended in CDK's Aromaticity/Cycles documentation
//      (Daylight electron donation, all cycles with a fallback to cycles of size <= 6
//      when the full cycle set is intractable). CDK's own descriptor unit tests mostly
//      use the deprecated Aromaticity.cdkLegacy(); descriptors that need that model
//      (BCUT, AcidicGroupCount, ATSp) re-perceive it themselves by default, so the
//      choice here only affects descriptors that read the atom aromatic flag as-is
//      (LargestPiSystem).
// Every descriptor runs with its default parameters on its own clone of the
// prepared molecule (some descriptors modify their input).
//
// Cell conventions: floats use Double.toString (NaN -> nan, +/-Infinity -> inf/-inf).
// If CDK reports a calculation error (DescriptorValue.getException() != null, CDK then
// returns a dummy value such as NaN, -1 or 0) every cell of that descriptor is `nan`.
// If the descriptor throws, its cells are left empty. Rows CDK cannot parse/prepare
// have `ERR` in the first value column and empty cells elsewhere.
//
// PubChem fingerprint: prepared as the PubchemFingerprinter javadoc prescribes, on a
// clone: percieveAtomTypesAndConfigureAtoms, Aromaticity.cdkLegacy().apply,
// convertImplicitToExplicitHydrogens; then getBitFingerprint (881 bits).

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.BitSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import org.openscience.cdk.CDK;
import org.openscience.cdk.aromaticity.Aromaticity;
import org.openscience.cdk.aromaticity.ElectronDonation;
import org.openscience.cdk.fingerprint.PubchemFingerprinter;
import org.openscience.cdk.graph.Cycles;
import org.openscience.cdk.interfaces.IAtomContainer;
import org.openscience.cdk.interfaces.IChemObjectBuilder;
import org.openscience.cdk.qsar.DescriptorValue;
import org.openscience.cdk.qsar.IMolecularDescriptor;
import org.openscience.cdk.qsar.descriptors.molecular.APolDescriptor;
import org.openscience.cdk.qsar.descriptors.molecular.AcidicGroupCountDescriptor;
import org.openscience.cdk.qsar.descriptors.molecular.AutocorrelationDescriptorCharge;
import org.openscience.cdk.qsar.descriptors.molecular.AutocorrelationDescriptorMass;
import org.openscience.cdk.qsar.descriptors.molecular.AutocorrelationDescriptorPolarizability;
import org.openscience.cdk.qsar.descriptors.molecular.BCUTDescriptor;
import org.openscience.cdk.qsar.descriptors.molecular.BPolDescriptor;
import org.openscience.cdk.qsar.descriptors.molecular.BasicGroupCountDescriptor;
import org.openscience.cdk.qsar.descriptors.molecular.CarbonTypesDescriptor;
import org.openscience.cdk.qsar.descriptors.molecular.ChiChainDescriptor;
import org.openscience.cdk.qsar.descriptors.molecular.ChiClusterDescriptor;
import org.openscience.cdk.qsar.descriptors.molecular.ChiPathClusterDescriptor;
import org.openscience.cdk.qsar.descriptors.molecular.ChiPathDescriptor;
import org.openscience.cdk.qsar.descriptors.molecular.EccentricConnectivityIndexDescriptor;
import org.openscience.cdk.qsar.descriptors.molecular.FragmentComplexityDescriptor;
import org.openscience.cdk.qsar.descriptors.molecular.HybridizationRatioDescriptor;
import org.openscience.cdk.qsar.descriptors.molecular.KappaShapeIndicesDescriptor;
import org.openscience.cdk.qsar.descriptors.molecular.LargestChainDescriptor;
import org.openscience.cdk.qsar.descriptors.molecular.LargestPiSystemDescriptor;
import org.openscience.cdk.qsar.descriptors.molecular.LongestAliphaticChainDescriptor;
import org.openscience.cdk.qsar.descriptors.molecular.MDEDescriptor;
import org.openscience.cdk.qsar.descriptors.molecular.PetitjeanNumberDescriptor;
import org.openscience.cdk.qsar.descriptors.molecular.SmallRingDescriptor;
import org.openscience.cdk.qsar.descriptors.molecular.VAdjMaDescriptor;
import org.openscience.cdk.qsar.descriptors.molecular.WeightedPathDescriptor;
import org.openscience.cdk.qsar.descriptors.molecular.WienerNumbersDescriptor;
import org.openscience.cdk.qsar.descriptors.molecular.ZagrebIndexDescriptor;
import org.openscience.cdk.qsar.result.DoubleArrayResult;
import org.openscience.cdk.qsar.result.DoubleResult;
import org.openscience.cdk.qsar.result.IDescriptorResult;
import org.openscience.cdk.qsar.result.IntegerArrayResult;
import org.openscience.cdk.qsar.result.IntegerResult;
import org.openscience.cdk.silent.SilentChemObjectBuilder;
import org.openscience.cdk.smiles.SmilesParser;
import org.openscience.cdk.tools.CDKHydrogenAdder;
import org.openscience.cdk.tools.manipulator.AtomContainerManipulator;

public final class CdkParityReference {

    private static final String SCRIPT = "scripts/cdk_reference/CdkParityReference.java";
    private static final String DATA = "crates/ferrum_core/tests/data";
    private static final String FP_PUBCHEM = "fp:PubChem";

    /** A CDK descriptor and the column names its outputs are written under. */
    private static final class Spec {
        final IMolecularDescriptor descriptor;
        final List<String> columns = new ArrayList<>();
        int calls, cdkErrors, thrown;
        long nanos;

        /**
         * @param rename CDK output name -> FerrumChem column (others keep the CDK name)
         * @param prefix prefix for every column ("" for none)
         */
        Spec(IMolecularDescriptor descriptor, IChemObjectBuilder bldr, String prefix, Map<String, String> rename) {
            this.descriptor = descriptor;
            descriptor.initialise(bldr);
            for (String name : descriptor.getDescriptorNames())
                columns.add(prefix + rename.getOrDefault(name, name));
        }

        String name() {
            return descriptor.getClass().getSimpleName();
        }
    }

    private static Map<String, String> rename(String... pairs) {
        Map<String, String> m = new LinkedHashMap<>();
        for (int i = 0; i < pairs.length; i += 2)
            m.put(pairs[i], pairs[i + 1]);
        return m;
    }

    private static List<Spec> specs(IChemObjectBuilder b) {
        Map<String, String> none = rename();
        List<Spec> s = new ArrayList<>();
        s.add(new Spec(new SmallRingDescriptor(), b, "", none));
        s.add(new Spec(new CarbonTypesDescriptor(), b, "", none));
        s.add(new Spec(new LargestPiSystemDescriptor(), b, "", none));
        s.add(new Spec(new LargestChainDescriptor(), b, "", none));
        s.add(new Spec(new LongestAliphaticChainDescriptor(), b, "", none));
        s.add(new Spec(new PetitjeanNumberDescriptor(), b, "", none));
        s.add(new Spec(new WienerNumbersDescriptor(), b, "",
                       rename("WPATH", "WienerPath", "WPOL", "WienerPolarity")));
        s.add(new Spec(new ZagrebIndexDescriptor(), b, "", rename("Zagreb", "Zagreb1")));
        s.add(new Spec(new APolDescriptor(), b, "", none));
        s.add(new Spec(new BPolDescriptor(), b, "", none));
        s.add(new Spec(new AcidicGroupCountDescriptor(), b, "", rename("nAcid", "nAcidicGroups")));
        s.add(new Spec(new BasicGroupCountDescriptor(), b, "", rename("nBase", "nBasicGroups")));
        s.add(new Spec(new HybridizationRatioDescriptor(), b, "", none));
        s.add(new Spec(new VAdjMaDescriptor(), b, "", none));
        s.add(new Spec(new FragmentComplexityDescriptor(), b, "", none));
        s.add(new Spec(new AutocorrelationDescriptorMass(), b, "", none));
        s.add(new Spec(new AutocorrelationDescriptorPolarizability(), b, "", none));
        s.add(new Spec(new AutocorrelationDescriptorCharge(), b, "", none));
        s.add(new Spec(new BCUTDescriptor(), b, "", none));
        s.add(new Spec(new MDEDescriptor(), b, "", none));
        s.add(new Spec(new WeightedPathDescriptor(), b, "", none));
        s.add(new Spec(new ChiPathDescriptor(), b, "", none));
        s.add(new Spec(new ChiClusterDescriptor(), b, "", none));
        s.add(new Spec(new ChiChainDescriptor(), b, "", none));
        s.add(new Spec(new ChiPathClusterDescriptor(), b, "", none));
        s.add(new Spec(new EccentricConnectivityIndexDescriptor(), b, "", none));
        s.add(new Spec(new KappaShapeIndicesDescriptor(), b, "", none));
        return s;
    }

    static String fmt(double x) {
        if (Double.isNaN(x)) return "nan";
        if (Double.isInfinite(x)) return x > 0 ? "inf" : "-inf";
        return Double.toString(x);
    }

    static String csv(String s) {
        if (s.indexOf(',') < 0 && s.indexOf('"') < 0 && s.indexOf('\n') < 0) return s;
        return '"' + s.replace("\"", "\"\"") + '"';
    }

    /** Flattens a descriptor result to its cells (one per output name). */
    static List<String> cells(IDescriptorResult r) {
        List<String> out = new ArrayList<>();
        if (r instanceof DoubleResult) {
            out.add(fmt(((DoubleResult) r).doubleValue()));
        } else if (r instanceof IntegerResult) {
            out.add(Integer.toString(((IntegerResult) r).intValue()));
        } else if (r instanceof DoubleArrayResult) {
            DoubleArrayResult a = (DoubleArrayResult) r;
            for (int i = 0; i < a.length(); i++) out.add(fmt(a.get(i)));
        } else if (r instanceof IntegerArrayResult) {
            IntegerArrayResult a = (IntegerArrayResult) r;
            for (int i = 0; i < a.length(); i++) out.add(Integer.toString(a.get(i)));
        } else {
            throw new IllegalStateException("unsupported result type " + r.getClass());
        }
        return out;
    }

    static void fill(List<String> row, int n, String v) {
        for (int i = 0; i < n; i++) row.add(v);
    }

    static IAtomContainer prepare(SmilesParser sp, CDKHydrogenAdder hadder, Aromaticity arom, String smi)
            throws Exception {
        IAtomContainer mol = sp.parseSmiles(smi);
        AtomContainerManipulator.percieveAtomTypesAndConfigureAtoms(mol);
        hadder.addImplicitHydrogens(mol);
        arom.apply(mol);
        return mol;
    }

    static String pubchem(PubchemFingerprinter fp, IAtomContainer prepared) throws Exception {
        IAtomContainer mol = prepared.clone();
        AtomContainerManipulator.percieveAtomTypesAndConfigureAtoms(mol);
        Aromaticity.cdkLegacy().apply(mol);
        AtomContainerManipulator.convertImplicitToExplicitHydrogens(mol);
        BitSet bits = fp.getBitFingerprint(mol).asBitSet();
        StringBuilder sb = new StringBuilder();
        for (int i = bits.nextSetBit(0); i >= 0; i = bits.nextSetBit(i + 1)) {
            if (sb.length() > 0) sb.append(' ');
            sb.append(i);
        }
        return sb.toString();
    }

    public static void main(String[] args) throws IOException {
        Path root = Paths.get(args.length > 0 ? args[0] : ".").toAbsolutePath().normalize();
        Path in = root.resolve(DATA).resolve("parity_molecules.smi");
        Path out = root.resolve(DATA).resolve("cdk_parity_reference.csv");

        IChemObjectBuilder bldr = SilentChemObjectBuilder.getInstance();
        SmilesParser sp = new SmilesParser(bldr);
        CDKHydrogenAdder hadder = CDKHydrogenAdder.getInstance(bldr);
        Aromaticity arom = new Aromaticity(ElectronDonation.daylight(), Cycles.or(Cycles.all(), Cycles.all(6)));
        PubchemFingerprinter pubchem = new PubchemFingerprinter(bldr);
        List<Spec> specs = specs(bldr);

        List<String> header = new ArrayList<>();
        header.add("smiles");
        for (Spec s : specs) header.addAll(s.columns);
        header.add(FP_PUBCHEM);
        final int nValues = header.size() - 1;

        long t0 = System.nanoTime();
        long fpNanos = 0;
        int n = 0, failed = 0, fpFailed = 0;
        try (BufferedReader src = Files.newBufferedReader(in, StandardCharsets.UTF_8);
             BufferedWriter dst = Files.newBufferedWriter(out, StandardCharsets.UTF_8)) {
            dst.write("# CDK " + CDK.getVersion() + " reference, generated by " + SCRIPT + "\n");
            dst.write(String.join(",", header));
            dst.write("\n");

            String line;
            while ((line = src.readLine()) != null) {
                String smi = line.trim();
                if (smi.isEmpty()) continue;
                n++;
                List<String> row = new ArrayList<>(header.size());
                row.add(csv(smi));

                IAtomContainer mol;
                try {
                    mol = prepare(sp, hadder, arom, smi);
                } catch (Exception | StackOverflowError e) {
                    failed++;
                    System.err.println("ERR " + smi + ": " + e);
                    row.add("ERR");
                    fill(row, nValues - 1, "");
                    dst.write(String.join(",", row));
                    dst.write("\n");
                    continue;
                }

                for (Spec s : specs) {
                    int k = s.columns.size();
                    long t = System.nanoTime();
                    s.calls++;
                    try {
                        DescriptorValue dv = s.descriptor.calculate(mol.clone());
                        if (dv.getException() != null) {
                            s.cdkErrors++;
                            fill(row, k, "nan");
                        } else {
                            List<String> c = cells(dv.getValue());
                            if (c.size() != k)
                                throw new IllegalStateException(c.size() + " values for " + k + " names");
                            row.addAll(c);
                        }
                    } catch (Exception | StackOverflowError e) {
                        s.thrown++;
                        System.err.println(s.name() + " threw on " + smi + ": " + e);
                        fill(row, k, "");
                    }
                    s.nanos += System.nanoTime() - t;
                }

                long t = System.nanoTime();
                try {
                    row.add(pubchem(pubchem, mol));
                } catch (Exception | StackOverflowError e) {
                    fpFailed++;
                    System.err.println("PubchemFingerprinter threw on " + smi + ": " + e);
                    row.add("");
                }
                fpNanos += System.nanoTime() - t;

                dst.write(String.join(",", row));
                dst.write("\n");
            }
        }

        double secs = (System.nanoTime() - t0) / 1e9;
        System.err.printf("wrote %s (%d molecules, %d not parsed/prepared by CDK) in %.1f s%n",
                          root.relativize(out), n, failed, secs);
        System.err.println("per descriptor: calls / CDK-reported errors (nan) / thrown (empty) / seconds");
        for (Spec s : specs)
            System.err.printf("  %-42s %5d %5d %5d %8.2f%n", s.name(), s.calls, s.cdkErrors, s.thrown, s.nanos / 1e9);
        System.err.printf("  %-42s %5d %5s %5d %8.2f%n", "PubchemFingerprinter", n - failed, "-", fpFailed,
                          fpNanos / 1e9);
    }
}
