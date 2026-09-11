import fs from 'node:fs/promises';
import path from 'node:path';
import { FileBlob, SpreadsheetFile } from '@oai/artifact-tool';
const root=path.resolve(process.argv[2]??'outputs/A题建模成果');
const qa=path.resolve(process.argv[3]??'work/excel_qa');
await fs.mkdir(qa,{recursive:true});
for (let n=1;n<=4;n++) {
 const name=`result${n}`;
 const sheets=JSON.parse(await fs.readFile(path.join(root,'results',name+'.json'),'utf8'));
 const w=await SpreadsheetFile.importXlsx(await FileBlob.load(path.join(root,'data','templates',name+'.xlsx')));
 for (const [sheetName,rows] of Object.entries(sheets)) {
  const s=w.worksheets.getItem(sheetName);
  s.getRange('A1:F5').clear({applyTo:'contents'});
  for(let start=0;start<rows.length;start+=1000) {
    const batch=rows.slice(start,start+1000);
    s.getRangeByIndexes(start,0,batch.length,rows[0].length).values=batch;
  }
  const used=s.getRangeByIndexes(0,0,rows.length,rows[0].length);
  used.format.font={name:'宋体',size:11};
  used.format.rowHeight=18;
  used.format.columnWidth=11;
  used.format.verticalAlignment='center';
  s.getRangeByIndexes(0,0,rows.length,1).format.columnWidth=34;
  s.getRangeByIndexes(0,0,1,rows[0].length).format.rowHeight=25;
  s.getRangeByIndexes(0,0,1,rows[0].length).format.horizontalAlignment='center';
  s.getRangeByIndexes(0,1,1,21).setNumberFormat('0.0');
  s.getRangeByIndexes(1,0,rows.length-1,1).setNumberFormat('0');
  s.getRangeByIndexes(1,1,rows.length-1,rows[0].length-1).setNumberFormat('0.0000');
  if(n===4){s.getRange('W1:X1').format.columnWidth=15;s.getRangeByIndexes(1,23,rows.length-1,1).setNumberFormat('0.000000');}
  s.freezePanes.freezeRows(1);
  s.freezePanes.freezeColumns(1);
 }
 w.recalculate();
 for(const [sheetName,rows] of Object.entries(sheets)) {
  const slug=n+'_'+(sheetName==='温度'?'T':sheetName==='水分浓度'?'C':'C');
  for (const [label,range] of [['first','A1:H9'],['last',`A${rows.length-6}:H${rows.length}`],['surface',n===4?'S1:X9':'S1:V9']]) {
    const p=await w.render({sheetName,range,scale:1.5,format:'png'});
    await fs.writeFile(path.join(qa,slug+'_'+label+'.png'),new Uint8Array(await p.arrayBuffer()));
  }
  const inspect=await w.inspect({kind:'table',range:`'${sheetName}'!A1:D4`,include:'values,formulas',tableMaxRows:4,tableMaxCols:4});
  await fs.writeFile(path.join(qa,slug+'_inspect.txt'),inspect.ndjson);
 }
 const errors=await w.inspect({kind:'match',searchTerm:'#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!',options:{useRegex:true,maxResults:20},summary:'formula error scan'});
 await fs.writeFile(path.join(qa,name+'_errors.txt'),errors.ndjson);
 const xlsx=await SpreadsheetFile.exportXlsx(w);
 await xlsx.save(path.join(root,name+'.xlsx'));
 console.log('SAVED',name,Object.entries(sheets).map(([s,r])=>[s,r.length-1,r[0].length]));
}
// This runtime sometimes leaves a worker open after successful exports.
process.exit(0);
