import com.comsol.model.*;
import com.comsol.model.util.*;
import java.io.*;
import java.util.*;

/**
 * 遗留事项 1：问题四 COMSOL 侧网格 / 时间步收敛研究
 * 空间：Nel = 40 / 160 / 640（rtol 固定 1e-4）
 * 时间：rtol = 1e-3 / 1e-4 / 1e-5（Nel 固定 160）
 */
public class Q4NoShrink {
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

    static void caseRun(String name, int nel, double rtol, double tend, double dtout) {
        try {
            Model m = ModelUtil.create(name);
            m.modelNode().create("comp1");
            m.param().set("hcv", "25");
            m.param().set("hms", "8e-7");
            m.param().set("tsw", "14400");
            m.param().set("Ta_inf", "49.99893443");
            m.param().set("Ca_inf", "0.04998754");
            m.geom().create("geom1", 1);
            m.geom("geom1").create("i1", "Interval");
            m.geom("geom1").feature("i1").set("p1", "0");
            m.geom("geom1").feature("i1").set("p2", "1");
            m.geom("geom1").run();
            mkfunc(m, "Ta_i", "D:/comsol_q4/data/Ta.txt");
            mkfunc(m, "Ca_i", "D:/comsol_q4/data/Ca.txt");
            mkfunc(m, "R_i", "D:/comsol_q4/data/R.txt");
            m.component("comp1").variable().create("var1");
            m.component("comp1").variable("var1").set("Ta_env",
                    "Ta_i(t)*(t<tsw)+Ta_inf*(t>=tsw)");
            m.component("comp1").variable("var1").set("Ca_env",
                    "Ca_i(t)*(t<tsw)+Ca_inf*(t>=tsw)");
            m.component("comp1").variable("var1").set("Rv", "0.02");   // 固定半径对照
            m.component("comp1").variable("var1").set("Da",
                    "4.2e-4*exp(-0.30/u)*exp(-3850/(u2+273.15))");
            m.component("comp1").variable("var1").set("rhoa", "760+90*u");
            m.component("comp1").variable("var1").set("cpa", "1850+2150*u/(u+1)");
            m.component("comp1").variable("var1").set("ka", "0.12+0.20*u/(u+1)");
            m.component("comp1").physics().create("cc", "CoefficientFormPDE", "geom1");
            m.component("comp1").physics("cc").prop("Units")
                    .set("DependentVariableQuantity", "none");
            m.component("comp1").physics("cc").feature("cfeq1").set("da", "x");
            m.component("comp1").physics("cc").feature("cfeq1").set("c", "x*Da/Rv^2");
            m.component("comp1").physics("cc").feature("cfeq1").set("f", "0");
            m.component("comp1").physics().create("ct", "CoefficientFormPDE", "geom1");
            m.component("comp1").physics("ct").prop("Units")
                    .set("DependentVariableQuantity", "none");
            m.component("comp1").physics("ct").feature("cfeq1").set("da", "x*rhoa*cpa");
            m.component("comp1").physics("ct").feature("cfeq1").set("c", "x*ka/Rv^2");
            m.component("comp1").physics("ct").feature("cfeq1").set("f", "0");
            m.component("comp1").physics("cc").feature("init1").set("u", "2.55");
            m.component("comp1").physics("ct").feature("init1").set("u2", "28");
            m.component("comp1").physics("cc").create("flux1", "FluxBoundary", 0);
            m.component("comp1").physics("cc").feature("flux1").selection().set(new int[]{2});
            m.component("comp1").physics("cc").feature("flux1").set("g", "-hms*(u-Ca_env)/Rv");
            m.component("comp1").physics("ct").create("flux1", "FluxBoundary", 0);
            m.component("comp1").physics("ct").feature("flux1").selection().set(new int[]{2});
            m.component("comp1").physics("ct").feature("flux1").set("g", "-hcv*(u2-Ta_env)/Rv");
            m.component("comp1").mesh().create("mesh1");
            m.component("comp1").mesh("mesh1").autoMeshSize(4);
            m.component("comp1").mesh("mesh1").run();
            m.component("comp1").mesh("mesh1").feature("size").set("custom", "on");
            m.component("comp1").mesh("mesh1").feature("size").set("hmax", "1.0/" + nel);
            m.component("comp1").mesh("mesh1").run();
            int nelAct = m.component("comp1").mesh("mesh1").stat().getNumElem();
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
            double el = (System.currentTimeMillis() - t0) / 1000.0;
            m.result().export().create("dF", "Data");
            m.result().export("dF").set("data", "dset1");
            m.result().export("dF").set("expr", new String[]{"u", "u2"});
            m.result().export("dF").set("location", "grid");
            m.result().export("dF").set("gridx1", "range(0,0.5,1)");
            m.result().export("dF").set("header", "off");
            m.result().export("dF").set("filename", OUT + "conv_" + name + ".csv");
            m.result().export("dF").run();
            p(String.format("%-10s nel=%-4d rtol=%-6s solve=%6.2f s  -> conv_%s.csv",
                    name, nelAct, String.valueOf(rtol), el, name));
        } catch (Throwable t) {
            p(name + " FAIL : " + t);
        }
    }

    public static Model run() {
        try {
            log = new PrintWriter(new OutputStreamWriter(
                    new FileOutputStream(OUT + "q4conv_log.txt"), "UTF-8"));
        } catch (Throwable t) { t.printStackTrace(); }
        caseRun("noshk", 160, 1e-4, 1296000, 600);
        try { log.close(); } catch (Throwable t) { }
        return null;
    }

    public static void main(String[] args) { run(); }
}

