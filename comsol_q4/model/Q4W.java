import com.comsol.model.*;
import com.comsol.model.util.*;
import java.io.*;
import java.util.*;

/**
 * 问题四：1D 平面几何 + 手动轴对称加权（等价于轴对称方程）
 *     x·∂u/∂t = ∂/∂x( x·(D/R²)·∂u/∂x )      ⇔  ∂u/∂t = 1/(R²x)·∂/∂x(x D ∂u/∂x)
 *     x·ρcp·∂θ/∂t = ∂/∂x( x·(k/R²)·∂θ/∂x )
 * 边界 x=1:  g = −hm(u−Ca)/R ,  g = −h(θ−Ta)/R
 */
public class Q4W {
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

    static void runCase(String name, int nel, double tend, double dtout, double rtol,
                        String daC, String cC, String daT, String cT,
                        String gC, String gT, String iniC, String iniT,
                        boolean withEnv) {
        try {
            Model m = ModelUtil.create(name);
            m.modelNode().create("comp1");
            m.param().set("hcv", "25");
            m.param().set("hms", "8e-7");
            m.param().set("tsw", "14400");
            m.param().set("Ta_inf", "49.99893443");
            m.param().set("Ca_inf", "0.04998754");
            m.param().set("R0", "0.02");
            m.geom().create("geom1", 1);       // 1D 平面（不加 axisymmetric）
            m.geom("geom1").create("i1", "Interval");
            m.geom("geom1").feature("i1").set("p1", "0");
            m.geom("geom1").feature("i1").set("p2", "1");
            m.geom("geom1").run();
            mkfunc(m, "Ta_i", "D:/comsol_q4/data/Ta.txt");
            mkfunc(m, "Ca_i", "D:/comsol_q4/data/Ca.txt");
            if (withEnv) mkfunc(m, "R_i", "D:/comsol_q4/data/R.txt");
            m.component("comp1").variable().create("var1");
            m.component("comp1").variable("var1").set("Ta_env",
                    withEnv ? "Ta_i(t)*(t<tsw)+Ta_inf*(t>=tsw)" : "Ta_i(t)");
            m.component("comp1").variable("var1").set("Ca_env",
                    withEnv ? "Ca_i(t)*(t<tsw)+Ca_inf*(t>=tsw)" : "Ca_i(t)");
            m.component("comp1").variable("var1").set("Rv", withEnv ? "R_i(t)" : "R0");
            m.component("comp1").variable("var1").set("Da",
                    "4.2e-4*exp(-0.30/u)*exp(-3850/(u2+273.15))");
            m.component("comp1").variable("var1").set("rhoa", "760+90*u");
            m.component("comp1").variable("var1").set("cpa", "1850+2150*u/(u+1)");
            m.component("comp1").variable("var1").set("ka", "0.12+0.20*u/(u+1)");

            m.component("comp1").physics().create("cc", "CoefficientFormPDE", "geom1");
            m.component("comp1").physics("cc").prop("Units")
                    .set("DependentVariableQuantity", "none");
            m.component("comp1").physics("cc").feature("cfeq1").set("da", daC);
            m.component("comp1").physics("cc").feature("cfeq1").set("c", cC);
            m.component("comp1").physics("cc").feature("cfeq1").set("f", "0");
            m.component("comp1").physics().create("ct", "CoefficientFormPDE", "geom1");
            m.component("comp1").physics("ct").prop("Units")
                    .set("DependentVariableQuantity", "none");
            m.component("comp1").physics("ct").feature("cfeq1").set("da", daT);
            m.component("comp1").physics("ct").feature("cfeq1").set("c", cT);
            m.component("comp1").physics("ct").feature("cfeq1").set("f", "0");
            m.component("comp1").physics("cc").feature("init1").set("u", iniC);
            m.component("comp1").physics("ct").feature("init1").set("u2", iniT);
            m.component("comp1").physics("cc").create("flux1", "FluxBoundary", 0);
            m.component("comp1").physics("cc").feature("flux1").selection().set(new int[]{2});
            m.component("comp1").physics("cc").feature("flux1").set("g", gC);
            m.component("comp1").physics("ct").create("flux1", "FluxBoundary", 0);
            m.component("comp1").physics("ct").feature("flux1").selection().set(new int[]{2});
            m.component("comp1").physics("ct").feature("flux1").set("g", gT);

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
            m.sol("sol1").feature("t1").set("rtol", String.valueOf(rtol));
            long t0 = System.currentTimeMillis();
            m.study("std1").run();
            p(name + " solve elapsed = " + (System.currentTimeMillis() - t0) / 1000.0 + " s");

            m.result().export().create("dF", "Data");
            m.result().export("dF").set("data", "dset1");
            m.result().export("dF").set("expr", new String[]{"u", "u2", "x"});
            m.result().export("dF").set("location", "grid");
            m.result().export("dF").set("gridx1", withEnv ? "range(0,0.002,1)" : "range(0,0.005,1)");
            m.result().export("dF").set("header", "off");
            m.result().export("dF").set("filename", OUT + "w_" + name + ".csv");
            m.result().export("dF").run();
            p(name + " exported size=" + new File(OUT + "w_" + name + ".csv").length());
        } catch (Throwable t) {
            p(name + " FAIL : " + t);
        }
    }

    public static Model run() {
        try {
            log = new PrintWriter(new OutputStreamWriter(
                    new FileOutputStream(OUT + "q4w_log.txt"), "UTF-8"));
        } catch (Throwable t) { t.printStackTrace(); }

        // 验证：问题一（附录2 常物性，R=0.02）
        runCase("valid", 400, 1800, 60, 1e-8,
                "x", "x*7e-9*exp(-0.89/u)/R0^2",
                "x*820*2600", "x*0.36/R0^2",
                "-hms*(u-Ca_env)/R0", "-hcv*(u2-Ta_env)/R0", "2.55", "28", false);

        // 问题四
        runCase("q4", 160, 345600, 300, 1e-4,
                "x", "x*Da/Rv^2",
                "x*rhoa*cpa", "x*ka/Rv^2",
                "-hms*(u-Ca_env)/Rv", "-hcv*(u2-Ta_env)/Rv", "2.55", "28", true);

        try { log.close(); } catch (Throwable t) { }
        return null;
    }

    public static void main(String[] args) { run(); }
}
