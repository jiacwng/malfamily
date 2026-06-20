// Ghidra headless post-script for malfamily.
//
// Runs INSIDE Ghidra after auto-analysis. It walks every recovered function and writes
// that function's name, entry virtual address, and the ordered list of
// instruction mnemonics to a JSON file.
//
// The output path is passed as the first script argument. Emitted shape:
//   {
//     "format":    "Portable Executable (PE)",
//     "processor": "x86",          // or "AARCH64", ...
//     "ptr_bytes": 8,              // 4 => 32-bit, 8 => 64-bit
//     "big_endian": false,
//     "functions": [ {"name": "...", "va": 123, "mnemonics": ["mov", ...]}, ... ]
//   }
//
// core/parser.py invokes this via analyzeHeadless and reads the JSON back.

import java.io.BufferedWriter;
import java.io.FileOutputStream;
import java.io.OutputStreamWriter;
import java.io.Writer;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.nio.file.StandardCopyOption;
import ghidra.app.script.GhidraScript;
import ghidra.program.model.listing.Function;
import ghidra.program.model.listing.FunctionManager;
import ghidra.program.model.listing.Instruction;
import ghidra.program.model.listing.InstructionIterator;
import ghidra.program.model.listing.Listing;

public class MalfamilyExport extends GhidraScript {
    @Override
    public void run() throws Exception {
        if (getScriptArgs().length < 1) {
            println("MalfamilyExport: missing output-path argument");
            return;
        }
        String outPath = getScriptArgs()[0];
        String partPath = outPath + ".part";
        FunctionManager fm = currentProgram.getFunctionManager();
        Listing listing = currentProgram.getListing();

        // Stream straight to disk as UTF-8: a binary with millions of instructions
        // never has to fit in a single in-memory string (avoids OutOfMemoryError),
        // and the charset is pinned to match parser.py's UTF-8 read.
        try (Writer w = new BufferedWriter(
                new OutputStreamWriter(new FileOutputStream(partPath), StandardCharsets.UTF_8))) {
            w.write("{");
            w.write("\"format\":" + js(currentProgram.getExecutableFormat()) + ",");
            w.write("\"processor\":" + js(currentProgram.getLanguage().getProcessor().toString()) + ",");
            w.write("\"ptr_bytes\":" + currentProgram.getDefaultPointerSize() + ",");
            w.write("\"big_endian\":" + currentProgram.getLanguage().isBigEndian() + ",");
            w.write("\"functions\":[");

            boolean firstFunc = true;
            for (Function f : fm.getFunctions(true)) {
                if (!firstFunc) w.write(",");
                firstFunc = false;
                w.write("{\"name\":" + js(f.getName()));
                w.write(",\"va\":" + f.getEntryPoint().getOffset());
                w.write(",\"mnemonics\":[");
                InstructionIterator it = listing.getInstructions(f.getBody(), true);
                boolean firstIns = true;
                while (it.hasNext()) {
                    Instruction ins = it.next();
                    if (!firstIns) w.write(",");
                    firstIns = false;
                    w.write(js(ins.getMnemonicString()));
                }
                w.write("]}");
            }
            w.write("]}");
        }

        // Publish atomically so parser.py only ever sees a fully-written file:
        // a crash before this point leaves only the .part file, which it treats as having no output.
        Files.move(Paths.get(partPath), Paths.get(outPath), StandardCopyOption.REPLACE_EXISTING);
        println("EXPORT-OK wrote JSON to " + outPath);
    }

    //JSON string escaper (avoids any external dependency).
    private static String js(String s) {
        if (s == null) return "\"\"";
        StringBuilder b = new StringBuilder("\"");
        for (int i = 0; i < s.length(); i++) {
            char c = s.charAt(i);
            switch (c) {
                case '\\': b.append("\\\\"); break;
                case '"': b.append("\\\""); break;
                case '\n': b.append("\\n"); break;
                case '\r': b.append("\\r"); break;
                case '\t': b.append("\\t"); break;
                default:
                    if (c < 0x20) b.append(String.format("\\u%04x", (int) c));
                    else b.append(c);
            }
        }
        return b.append("\"").toString();
    }
}
