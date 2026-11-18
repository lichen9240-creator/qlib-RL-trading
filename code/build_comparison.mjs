import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const project = path.dirname(path.dirname(fileURLToPath(import.meta.url)));
const outputDirectory = path.join(project, "outputs", "opds_comparison");
const results = JSON.parse(await fs.readFile(path.join(project, "results", "comparison.json"), "utf8"));
const csvText = (await fs.readFile(path.join(project, "results", "paired_orders.csv"), "utf8")).replace(/^\uFEFF/, "");
const imported = await Workbook.fromCSV(csvText, { sheetName: "Imported" });
const importedValues = imported.worksheets.getItemAt(0).getUsedRange().values;
const headers = importedValues[0];
const sourceRows = importedValues.slice(1).filter(row => row[0] !== null && row[0] !== "");
const columnIndex = Object.fromEntries(headers.map((header, index) => [header, index]));
const modelNames = [
  ["initialized", "训练前（重建初始化）", 0],
  ["validation_best", "训练后（验证集最佳）", 12],
  ["final", "训练后（最终模型）", 212],
  ["twap", "TWAP 基准", "规则"],
];
if (sourceRows.length !== 540) throw new Error(`Expected 540 matched orders, found ${sourceRows.length}`);

const workbook = Workbook.create();
const summary = workbook.worksheets.add("训练前后对比");
const details = workbook.worksheets.add("逐单明细");
const detailRows = [];
for (const [key, label] of modelNames) {
  for (const row of sourceRows) {
    const instrument = row[columnIndex.instrument];
    if (!/^(SH|SZ)\d{6}$/.test(instrument)) throw new Error(`Invalid instrument ${instrument}`);
    const numeric = suffix => {
      const raw = row[columnIndex[`${key}_${suffix}`]];
      if (raw === "" || raw === null || raw === undefined) throw new Error(`Missing ${key}_${suffix}`);
      const value = Number(raw);
      if (!Number.isFinite(value)) throw new Error(`Invalid ${key}_${suffix}`);
      return value;
    };
    detailRows.push([
      label,
      instrument,
      new Date(String(row[columnIndex.datetime]).slice(0, 10) + "T00:00:00Z"),
      numeric("pa_bps"),
      numeric("trade_cost"),
      numeric("deal_amount"),
      numeric("base_price"),
      null,
      null,
      null,
      numeric("ffr"),
      null,
    ]);
  }
}
const lastRow = detailRows.length + 1;
details.getRange("A1:L1").values = [[
  "模型", "股票", "交易日", "价格优势（bp）", "交易费用（元）", "成交数量（带方向）", "窗口基准价",
  "基准成交金额", "费用（bp）", "扣费后优势（bp）", "成交比例", "金额加权分子",
]];
details.getRange(`A2:L${lastRow}`).values = detailRows;
details.getRange("H2").formulas = [["=ABS(F2)*G2"]];
details.getRange(`H2:H${lastRow}`).fillDown();
details.getRange("I2").formulas = [["=E2/H2*10000"]];
details.getRange(`I2:I${lastRow}`).fillDown();
details.getRange("J2").formulas = [["=D2-I2"]];
details.getRange(`J2:J${lastRow}`).fillDown();
details.getRange("L2").formulas = [["=D2*H2"]];
details.getRange(`L2:L${lastRow}`).fillDown();
details.getRange(`A1:L${lastRow}`).format.font = { name: "Arial", size: 10, color: "#233044" };
details.getRange(`A1:L${lastRow}`).format.verticalAlignment = "center";
details.getRange(`A2:C${lastRow}`).format.horizontalAlignment = "left";
details.getRange(`D2:L${lastRow}`).format.horizontalAlignment = "right";
details.getRange("A1:L1").format = {
  fill: "#263E59",
  font: { name: "Arial", size: 10, color: "#FFFFFF", bold: true },
  horizontalAlignment: "center",
  verticalAlignment: "center",
  rowHeight: 26,
};
details.getRange("A1:A2161").format.columnWidth = 27;
details.getRange("B1:B2161").format.columnWidth = 13;
details.getRange("C1:C2161").format.columnWidth = 14;
details.getRange("D1:L2161").format.columnWidth = 18;
details.getRange(`C2:C${lastRow}`).setNumberFormat("yyyy-mm-dd");
details.getRange(`D2:J${lastRow}`).setNumberFormat("#,##0.0000;[Red](#,##0.0000)");
details.getRange(`K2:K${lastRow}`).setNumberFormat("0.00%");
details.getRange(`L2:L${lastRow}`).setNumberFormat("#,##0.0000;[Red](#,##0.0000)");
details.getRange(`A2:L${lastRow}`).format.rowHeight = 19;
details.getRange("N1").values = [["同一批 540 笔测试订单的官方 Qlib 嵌套回测。"]];
details.getRange("N2").values = [["原始输入保留价格优势、交易费用、成交数量、窗口基准价和成交比例。"]];
details.getRange("N3").values = [["费用、扣费后优势、金额加权分子由本表公式计算。"]];
details.getRange("N4").values = [["https://github.com/microsoft/qlib/blob/v0.9.7/examples/rl_order_execution/README.md"]];
details.getRange("N1:N4").format.font = { name: "Arial", size: 10, color: "#5B6574" };
details.getRange("N1:N4").format.columnWidth = 95;
details.freezePanes.freezeRows(1);
details.freezePanes.freezeColumns(3);
details.showGridLines = false;

summary.getRange("B2").values = [["OPDS 训练前后对比"]];
summary.getRange("B4").values = [["9 只股票，540 笔订单，测试区间 2021-10-08 至 2021-12-30。"]];
summary.getRange("B5").values = [["价格与费用单位为 bp，1 bp = 0.01%。增量均对比训练前。"]];
summary.getRange("B7:J7").values = [[
  "模型", "轮数", "平均价格优势", "金额加权优势", "平均费用", "扣费后优势", "成交比例", "平均优势增量", "扣费后增量",
]];
summary.getRange("B8:C11").values = modelNames.map(([, label, iteration]) => [label, iteration]);
const modelRange = `'逐单明细'!$A$2:$A$${lastRow}`;
const detailRange = column => `'逐单明细'!$${column}$2:$${column}$${lastRow}`;
for (let index = 0; index < modelNames.length; index++) {
  const row = index + 8;
  summary.getRange(`D${row}:J${row}`).formulas = [[
    `=AVERAGEIFS(${detailRange("D")},${modelRange},$B${row})`,
    `=SUMIFS(${detailRange("L")},${modelRange},$B${row})/SUMIFS(${detailRange("H")},${modelRange},$B${row})`,
    `=AVERAGEIFS(${detailRange("I")},${modelRange},$B${row})`,
    `=AVERAGEIFS(${detailRange("J")},${modelRange},$B${row})`,
    `=AVERAGEIFS(${detailRange("K")},${modelRange},$B${row})`,
    `=D${row}-$D$8`,
    `=G${row}-$G$8`,
  ]];
}
summary.getRange("B14").values = [["训练前权重按原环境与 seed 42 重建，原训练未保存第 0 轮检查点。"]];
summary.getRange("B15").values = [["第 12 轮依据验证集选出，第 212 轮因连续 50 次验证未改善而提前停止。"]];
summary.getRange("B16").values = [["卖出费率 0.15%，每笔最低 5 元，整手单位 100 股。所有组使用同一批测试订单。"]];
summary.getRange("B17").values = [["均值按订单等权，金额加权优势按成交数量绝对值乘窗口基准价加权。"]];
summary.getRange("B18").values = [["单随机种子、9 只股票的数据范围。训练及验证扣除睡眠约 27 小时 14 分。"]];
summary.getRange("B2:J18").format.font = { name: "Arial", size: 11, color: "#233044" };
summary.getRange("B2:J18").format.verticalAlignment = "center";
summary.getRange("B2").format.font = { name: "Arial", size: 16, color: "#233044", bold: true };
summary.getRange("B2:J2").format.borders = { bottom: { style: "thin", color: "#A8B4C5" } };
summary.getRange("B2:J2").format.rowHeight = 30;
summary.getRange("B4:B5").format.font = { name: "Arial", size: 10, color: "#5B6574", italic: true };
summary.getRange("B14:B18").format.font = { name: "Arial", size: 10, color: "#5B6574" };
summary.getRange("B14:J18").format.rowHeight = 21;
summary.getRange("B7:J7").format = {
  fill: "#263E59",
  font: { name: "Arial", size: 11, color: "#FFFFFF", bold: true },
  horizontalAlignment: "center",
  verticalAlignment: "center",
  rowHeight: 31,
};
summary.getRange("B8:J11").format.rowHeight = 30;
summary.getRange("B9:J9").format.fill = "#ECF1F7";
summary.getRange("B11:J11").format.fill = "#F4F6F8";
summary.getRange("C8:J11").format.horizontalAlignment = "right";
summary.getRange("D8:G11").setNumberFormat("0.0000;[Red](0.0000)");
summary.getRange("H8:H11").setNumberFormat("0.00%");
summary.getRange("I8:J11").setNumberFormat("+0.0000;[Red](0.0000);0.0000");
summary.getRange("C8:C11").setNumberFormat("0");
summary.getRange("A1:A18").format.columnWidth = 3;
summary.getRange("B1:B18").format.columnWidth = 29;
summary.getRange("C1:C18").format.columnWidth = 8;
summary.getRange("D1:G18").format.columnWidth = 17;
summary.getRange("H1:H18").format.columnWidth = 14;
summary.getRange("I1:J18").format.columnWidth = 18;
summary.showGridLines = false;
summary.tabColor = "#263E59";

workbook.recalculate();
const values = summary.getRange("D8:J11").values;
const expectedKeys = ["mean_pa_bps", "notional_weighted_pa_bps", "mean_fee_bps", "mean_fee_adjusted_pa_bps", "mean_fulfillment_ratio", "mean_pa_change_vs_initialized_bps", "mean_net_pa_change_vs_initialized_bps"];
for (let row = 0; row < modelNames.length; row++) {
  for (let column = 0; column < expectedKeys.length; column++) {
    const expected = results.models[modelNames[row][0]][expectedKeys[column]];
    const actual = Number(values[row][column]);
    if (!Number.isFinite(expected) || !Number.isFinite(actual) || Math.abs(actual - expected) > 1e-8) {
      throw new Error(`Summary mismatch ${modelNames[row][0]} ${expectedKeys[column]} ${values[row][column]} versus ${expected}`);
    }
  }
}
const inspection = await workbook.inspect({ kind: "table", range: "'训练前后对比'!B7:J11", include: "values,formulas", tableMaxRows: 5, tableMaxCols: 9, maxChars: 2500 });
console.log(inspection.ndjson);
const errors = await workbook.inspect({ kind: "match", searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!", options: { useRegex: true, maxResults: 20 }, summary: "formula error scan" });
console.log(errors.ndjson);
await fs.mkdir(outputDirectory, { recursive: true });
for (const [sheetName, range, filename] of [["训练前后对比", "A1:K19", "comparison_preview.png"], ["逐单明细", "A1:L12", "details_preview.png"]]) {
  const preview = await workbook.render({ sheetName, range, scale: 1.5, format: "png" });
  await fs.writeFile(path.join(outputDirectory, filename), new Uint8Array(await preview.arrayBuffer()));
}
const output = await SpreadsheetFile.exportXlsx(workbook);
await output.save(path.join(outputDirectory, "comparison.xlsx"));
console.log(JSON.stringify({ output: path.join(outputDirectory, "comparison.xlsx"), matchedOrders: sourceRows.length, detailRows: detailRows.length, summaryValues: values }));
