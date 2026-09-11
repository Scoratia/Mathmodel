import com.comsol.model.*;
import com.comsol.model.util.*;
import java.io.*;
import java.util.*;

/**
 * 前两问：按题目要求的时间/径向分辨率重算
 *   P1  问题1：附录2 物性，固定半径 2 cm，0—1800 s，每 1 s
 *   P2  问题2：附录3 物性，固定半径 2 cm，0—10800 s（3 h），每 1 s
 * 径向网格：x = r/R0，取 0,0.05,...,1.0 共 21 点（对应 r = 0,0.1,...,2.0 cm）
 */
public class Q4P12 {
    static PrintWriter log;
    static final String OUT = "D:/comsol_q4/out/";

    static void p(String s) {
        try { log.println(s); log.flush(); } catch (Throwable t) { }
        System.out.println(s);
    }

    static String[][] read2(String path) throws IOException {
        BufferedReader br = new BufferedReader(new FileReader(path));
        List rows = new ArrayList();
        String line;
        while ((line = br.readLine()) != null) {
            line = line.trim();
            if (line.length() == 0) continue;
            String[] parts = line.split("[\\s,]+");
            if (parts.length < 2) continue;
            rows.add(new String[]{parts[0], parts[1]});
        }
        br.close();
        return (String[][]) rows.toArray(new String[rows.size()][]);
    }

    static void mkfunc(Model m, String tag, String path) throws IOException {
        String[][] tab = read2(path);
        m.func().create(tag, "Interpolation");
        m.func(tag).set("source", "table");
        m.func(tag).set("nargs", 1);
        m.func(tag).set("table", tab);
        m.func(tag).set("extrap", "const");
        m.func(tag).set("funcname", tag);
    }

    static void caseRun(String name, String daC, String daT, String cT,
                        double tend, double dtout, int nel, double[] texp) {
        try {
            Model m = ModelUtil.create(name);
            m.modelNode().create("comp1");
            m.param().set("hcv", "25");
            m.param().set("hms", "8e-7");
            m.param().set("R0", "0.02");
            m.geom().create("geom1", 1);
            m.geom("geom1").create("i1", "Interval");
            m.geom("geom1").feature("i1").set("p1", "0");
            m.geom("geom1").feature("i1").set("p2", "1");
            m.geom("geom1").run();
            mkfunc(m, "Ta_i", "D:/comsol_q4/data/Ta.txt");
            mkfunc(m, "Ca_i", "D:/comsol_q4/data/Ca.txt");
            m.component("comp1").variable().create("var1");
            m.component("comp1").variable("var1").set("Ta_env", "Ta_i(t)");
            m.component("comp1").variable("var1").set("Ca_env", "Ca_i(t)");
            m.component("comp1").variable("var1").set("Rv", "R0");
            if (name.equals("p2")) {
                m.component("comp1").variable("var1").set("rho3", "650+128*u");
                m.component("comp1").variable("var1").set("cp3", "1450+2736*u/(u+1)");
                m.component("comp1").variable("var1").set("k3", "0.21+0.38*u/(u+1)");
            }
            m.component("comp1").physics().create("cc", "CoefficientFormPDE", "geom1");
            m.component("comp1").physics("cc").prop("Units")
                    .set("DependentVariableQuantity", "none");
            m.component("comp1").physics("cc").feature("cfeq1").set("da", "x");
            m.component("comp1").physics("cc").feature("cfeq1").set("c", daC);
            m.component("comp1").physics("cc").feature("cfeq1").set("f", "0");
            m.component("comp1").physics().create("ct", "CoefficientFormPDE", "geom1");
            m.component("comp1").physics("ct").prop("Units")
                    .set("DependentVariableQuantity", "none");
            m.component("comp1").physics("ct").feature("cfeq1").set("da", daT);
            m.component("comp1").physics("ct").feature("cfeq1").set("c", cT);
            m.component("comp1").physics("ct").feature("cfeq1").set("f", "0");
            m.component("comp1").physics("cc").feature("init1").set("u", "2.55");
            m.component("comp1").physics("ct").feature("init1").set("u2", "28");
            m.component("comp1").physics("cc").create("flux1", "FluxBoundary", 0);
            m.component("comp1").physics("cc").feature("flux1").selection().set(new int[]{2});
            m.component("comp1").physics("cc").feature("flux1").set("g", "-hms*(u-Ca_env)/R0");
            m.component("comp1").physics("ct").create("flux1", "FluxBoundary", 0);
            m.component("comp1").physics("ct").feature("flux1").selection().set(new int[]{2});
            m.component("comp1").physics("ct").feature("flux1").set("g", "-hcv*(u2-Ta_env)/R0");
            m.component("comp1").mesh().create("mesh1");
            m.component("comp1").mesh("mesh1").autoMeshSize(4);
            m.component("comp1").mesh("mesh1").run();
            m.component("comp1").mesh("mesh1").feature("size").set("custom", "on");
            m.component("comp1").mesh("mesh1").feature("size").set("hmax", "1.0/" + nel);
            m.component("comp1").mesh("mesh1").run();
            p(name + " elements = " + m.component("comp1").mesh("mesh1").stat().getNumElem());

            m.param().set("tend", String.valueOf(tend));
            m.param().set("dtout", String.valueOf(dtout));
            m.study().create("std1");
            m.study("std1").create("time", "Transient");
            m.study("std1").feature("time").set("tlist", "range(0,dtout,tend)");
            m.study("std1").createAutoSequences("sol");
            m.sol("sol1").feature("t1").feature("fc1").set("maxiter", "60");
            m.sol("sol1").feature("t1").set("rtol", "1e-8");
            long t0 = System.currentTimeMillis();
            m.study("std1").run();
            p(name + " solve elapsed = " + (System.currentTimeMillis() - t0) / 1000.0 + " s");

            m.result().export().create("dF", "Data");
            m.result().export("dF").set("data", "dset1");
            m.result().export("dF").set("expr", new String[]{"u", "u2"});
            m.result().export("dF").set("location", "grid");
            m.result().export("dF").set("gridx1", "range(0,0.05,1)");
            m.result().export("dF").set("header", "off");
            m.result().export("dF").set("t", texp);
            m.result().export("dF").set("filename", OUT + "tab_" + name + ".csv");
            m.result().export("dF").run();
            p(name + " exported size=" + new File(OUT + "tab_" + name + ".csv").length());
        } catch (Throwable t) {
            p(name + " FAIL : " + t);
        }
    }

    public static Model run() {
        try {
            log = new PrintWriter(new OutputStreamWriter(
                    new FileOutputStream(OUT + "q4p12_log.txt"), "UTF-8"));
        } catch (Throwable t) { t.printStackTrace(); }
        p("=== 问题1：附录2 物性，固定半径，0—1800 s，步长 1 s ===");
        caseRun("p1", "x*7e-9*exp(-0.89/u)/R0^2", "x*820*2600", "x*0.36/R0^2",
                1800, 1, 400, new double[]{100, 300, 600, 900, 1200, 1500, 1800});
        p("=== 问题2：附录3 物性，固定半径，0—10800 s，步长 1 s ===");
        caseRun("p2", "x*2.4e-3*exp(-0.45/u)*exp(-3850/(u2+273.15))/R0^2",
                "x*rho3*cp3", "x*k3/R0^2", 10800, 1, 400,
                new double[]{1800, 3600, 5400, 7200, 9000, 10800});
        try { log.close(); } catch (Throwable t) { }
        return null;
    }

    public static void main(String[] args) { run(); }
}
