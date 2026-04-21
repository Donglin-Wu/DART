"""
Interactive GUI for dual-fidelity sequential fusion.
"""

import ctypes
import logging
import os
import queue
import threading
import traceback
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from base import logger
from predict_and_plot_from_pkl import ComputeMetrics, PredictRows
from sequential import SequentialFusion
from visualization import PredictFusionRows


BEST_KRG_CORR = "matern52"
BEST_MODEL_CONFIG = {
    "low": {
        "corr": "matern52",
        "poly_type": "constant",
        "nugget_val": 1e-3,
        "l_min": 0.25,
        "l_max": 10.0,
    },
    "delta": {
        "corr": "matern52",
        "poly_type": "linear",
        "nugget_val": 1e-8,
        "l_min": 0.25,
        "l_max": 10.0,
    },
}


class TkLogHandler(logging.Handler):
    def __init__(self, app):
        super().__init__(level=logging.INFO)
        self.app = app
        self.setFormatter(logging.Formatter("%(message)s"))

    def emit(self, record):
        message = self.format(record)
        self.app.PostUiEvent({"type": "log", "message": message})


class GuiInteractor:
    def __init__(self, app):
        self.app = app
        self.stopEvent = threading.Event()
        self.initialCsvEvent = threading.Event()
        self.initialCsvPath = None
        self.pointEvent = threading.Event()
        self.pointResponses = None

    def ShowInitialPoints(self, points):
        self.app.PostUiEvent({"type": "initial_points", "points": [np.asarray(point, dtype=float) for point in points]})

    def RequestInitialHighCsv(self, points):
        self.initialCsvPath = None
        self.initialCsvEvent.clear()
        self.app.PostUiEvent({"type": "request_initial_csv", "points": [np.asarray(point, dtype=float) for point in points]})
        self.initialCsvEvent.wait()
        return self.initialCsvPath

    def SubmitInitialHighCsv(self, filePath):
        self.initialCsvPath = filePath
        self.initialCsvEvent.set()

    def ShowRecommendedPoints(self, points):
        self.app.PostUiEvent({"type": "recommended_points", "points": [np.asarray(point, dtype=float) for point in points]})

    def RequestPointValues(self, points, dim, isInitial, xHigh, yHigh):
        self.pointResponses = None
        self.pointEvent.clear()
        self.app.PostUiEvent(
            {
                "type": "request_points",
                "points": [np.asarray(point, dtype=float) for point in points],
                "dim": dim,
                "isInitial": isInitial,
            }
        )
        self.pointEvent.wait()
        responseList = [] if self.pointResponses is None else list(self.pointResponses)
        return responseList, self.stopEvent.is_set()

    def SubmitPointResponses(self, responseList):
        self.pointResponses = responseList
        self.pointEvent.set()

    def NotifyInputError(self, message):
        self.app.PostUiEvent({"type": "input_error", "message": message})

    def UpdateState(self, stateData):
        self.app.PostUiEvent({"type": "state", "state": stateData})

    def ShouldStop(self):
        return self.stopEvent.is_set()

    def RequestStop(self):
        self.stopEvent.set()
        self.initialCsvEvent.set()
        self.pointEvent.set()


class FusionGuiApp:
    def __init__(self, root):
        self.root = root
        self.ConfigureDisplay()
        self.root.title("Dual-fidelity Sequential Fusion GUI")
        self.root.geometry("1820x1080")

        self.baseDir = os.path.dirname(os.path.abspath(__file__))
        self.uiQueue = queue.Queue()
        self.interactor = None
        self.workerThread = None
        self.logHandler = None
        self.latestState = None
        self.currentPointCount = 0
        self.isRunning = False
        self.isAutoSubmittingPoints = False

        self.BuildLayout()
        self.root.after(100, self.ProcessUiQueue)

    def ConfigureDisplay(self):
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except Exception:
            try:
                ctypes.windll.user32.SetProcessDPIAware()
            except Exception:
                pass

        try:
            screenDpi = self.root.winfo_fpixels("1i")
            scaleValue = max(1.25, min(screenDpi / 72.0, 2.2))
            self.root.tk.call("tk", "scaling", scaleValue)
        except Exception:
            pass

        try:
            defaultFont = ("Microsoft YaHei UI", 11)
            self.root.option_add("*Font", defaultFont)
            self.root.option_add("*TCombobox*Listbox*Font", defaultFont)
        except Exception:
            pass

    def BuildLayout(self):
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_rowconfigure(1, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_columnconfigure(1, weight=1)

        self.inputFrame = ttk.LabelFrame(self.root, text="User Input")
        self.inputFrame.grid(row=0, column=0, sticky="nsew", padx=8, pady=8)
        self.logFrame = ttk.LabelFrame(self.root, text="Program Console Output")
        self.logFrame.grid(row=1, column=0, sticky="nsew", padx=8, pady=8)
        self.fusionFrame = ttk.LabelFrame(self.root, text="Real-time Fusion Visualization")
        self.fusionFrame.grid(row=0, column=1, sticky="nsew", padx=8, pady=8)
        self.scatterFrame = ttk.LabelFrame(self.root, text="Real-time Regression Fit Scatter Plot")
        self.scatterFrame.grid(row=1, column=1, sticky="nsew", padx=8, pady=8)

        self.BuildInputFrame()
        self.BuildLogFrame()
        self.BuildFusionFrame()
        self.BuildScatterFrame()

    def BuildInputFrame(self):
        self.inputFrame.grid_columnconfigure(1, weight=1)

        ttk.Label(self.inputFrame, text="Low-fidelity Data CSV").grid(row=0, column=0, sticky="w", padx=6, pady=6)
        self.lowCsvVar = tk.StringVar()
        ttk.Entry(self.inputFrame, textvariable=self.lowCsvVar).grid(row=0, column=1, sticky="ew", padx=6, pady=6)
        ttk.Button(self.inputFrame, text="Browse", command=lambda: self.SelectFile(self.lowCsvVar)).grid(row=0, column=2, padx=6, pady=6)

        ttk.Label(self.inputFrame, text="High-fidelity Validation CSV").grid(row=1, column=0, sticky="w", padx=6, pady=6)
        self.validationCsvVar = tk.StringVar()
        ttk.Entry(self.inputFrame, textvariable=self.validationCsvVar).grid(row=1, column=1, sticky="ew", padx=6, pady=6)
        ttk.Button(self.inputFrame, text="Browse", command=lambda: self.SelectFile(self.validationCsvVar)).grid(row=1, column=2, padx=6, pady=6)

        ttk.Label(self.inputFrame, text="BHF").grid(row=2, column=0, sticky="w", padx=6, pady=6)
        self.bhfVar = tk.StringVar()
        ttk.Entry(self.inputFrame, textvariable=self.bhfVar).grid(row=2, column=1, sticky="ew", padx=6, pady=6)

        buttonFrame = ttk.Frame(self.inputFrame)
        buttonFrame.grid(row=3, column=0, columnspan=3, sticky="ew", padx=6, pady=6)
        buttonFrame.grid_columnconfigure(0, weight=1)
        buttonFrame.grid_columnconfigure(1, weight=1)
        self.startButton = ttk.Button(buttonFrame, text="Start", command=self.StartRun)
        self.startButton.grid(row=0, column=0, sticky="ew", padx=4)
        self.stopButton = ttk.Button(buttonFrame, text="Stop Early", command=self.RequestStop, state="disabled")
        self.stopButton.grid(row=0, column=1, sticky="ew", padx=4)

        self.initialFrame = ttk.LabelFrame(self.inputFrame, text="Initial High-fidelity Data Submission")
        self.initialFrame.grid(row=4, column=0, columnspan=3, sticky="nsew", padx=6, pady=8)
        self.initialFrame.grid_columnconfigure(0, weight=1)
        self.initialFrame.grid_columnconfigure(1, weight=1)
        self.initialPointText = tk.Text(self.initialFrame, height=7, wrap="word")
        self.initialPointText.grid(row=0, column=0, columnspan=3, sticky="nsew", padx=6, pady=6)

        self.initialCsvVar = tk.StringVar()
        ttk.Entry(self.initialFrame, textvariable=self.initialCsvVar).grid(row=1, column=0, sticky="ew", padx=6, pady=6)
        self.initialBrowseButton = ttk.Button(self.initialFrame, text="Browse Initial High-fidelity CSV", command=lambda: self.SelectFile(self.initialCsvVar), state="disabled")
        self.initialBrowseButton.grid(row=1, column=1, padx=6, pady=6)
        self.initialSubmitButton = ttk.Button(self.initialFrame, text="Submit Initial High-fidelity Data", command=self.SubmitInitialCsv, state="disabled")
        self.initialSubmitButton.grid(row=1, column=2, padx=6, pady=6)

        self.pointFrame = ttk.LabelFrame(self.inputFrame, text="Recommended Point Input During Iteration")
        self.pointFrame.grid(row=5, column=0, columnspan=3, sticky="nsew", padx=6, pady=8)
        self.pointFrame.grid_columnconfigure(1, weight=1)

        self.pointLabelVars = [tk.StringVar(value="Point 1: waiting for recommendation"), tk.StringVar(value="Point 2: waiting for recommendation")]
        self.pointEntryVars = [tk.StringVar(), tk.StringVar()]
        self.pointEntries = []
        self.pointSkipButtons = []
        self.pointConfirmButtons = []
        self.pointStatuses = [None, None]

        for pointIndex in range(2):
            ttk.Label(self.pointFrame, textvariable=self.pointLabelVars[pointIndex]).grid(row=pointIndex, column=0, sticky="w", padx=6, pady=6)
            entryWidget = ttk.Entry(self.pointFrame, textvariable=self.pointEntryVars[pointIndex], state="disabled")
            entryWidget.grid(row=pointIndex, column=1, sticky="ew", padx=6, pady=6)
            skipButton = ttk.Button(self.pointFrame, text="Skip", command=lambda index=pointIndex: self.ToggleSkip(index), state="disabled")
            skipButton.grid(row=pointIndex, column=2, padx=6, pady=6)
            confirmButton = ttk.Button(self.pointFrame, text="Confirm", command=lambda index=pointIndex: self.ConfirmPoint(index), state="disabled")
            confirmButton.grid(row=pointIndex, column=3, padx=6, pady=6)
            self.pointEntries.append(entryWidget)
            self.pointSkipButtons.append(skipButton)
            self.pointConfirmButtons.append(confirmButton)

        self.pointHintLabel = ttk.Label(self.pointFrame, text="Input format: enter only y, or x1,...,xd,y. Confirm or skip each point separately.")
        self.pointHintLabel.grid(row=2, column=0, columnspan=4, sticky="w", padx=6, pady=6)
        self.pointSubmitButton = ttk.Button(self.pointFrame, text="Submit Current Round Inputs", command=self.SubmitPointInputs, state="disabled")
        self.pointSubmitButton.grid(row=3, column=0, columnspan=4, sticky="ew", padx=6, pady=8)

    def BuildLogFrame(self):
        self.logFrame.grid_rowconfigure(0, weight=1)
        self.logFrame.grid_columnconfigure(0, weight=1)
        self.logText = tk.Text(self.logFrame, wrap="word")
        self.logText.grid(row=0, column=0, sticky="nsew", padx=6, pady=6)
        logScrollbar = ttk.Scrollbar(self.logFrame, orient="vertical", command=self.logText.yview)
        logScrollbar.grid(row=0, column=1, sticky="ns", pady=6)
        self.logText.configure(yscrollcommand=logScrollbar.set)

    def BuildFusionFrame(self):
        optionFrame = ttk.Frame(self.fusionFrame)
        optionFrame.pack(fill="x", padx=6, pady=6)

        self.showLowVar = tk.BooleanVar(value=True)
        self.showHighVar = tk.BooleanVar(value=True)
        self.showFusionVar = tk.BooleanVar(value=True)
        self.showValidationVar = tk.BooleanVar(value=True)
        self.showConfidenceVar = tk.BooleanVar(value=True)

        for textValue, variable in [
            ("Low Fidelity", self.showLowVar),
            ("High Fidelity", self.showHighVar),
            ("Fusion Model", self.showFusionVar),
            ("Validation Data", self.showValidationVar),
            ("Confidence Interval", self.showConfidenceVar),
        ]:
            ttk.Checkbutton(optionFrame, text=textValue, variable=variable, command=self.RedrawPlots).pack(side="left", padx=4)

        self.fusionFigure = Figure(figsize=(8.2, 5.2), dpi=140)
        self.fusionAxis = self.fusionFigure.add_subplot(111)
        self.fusionCanvas = FigureCanvasTkAgg(self.fusionFigure, master=self.fusionFrame)
        self.fusionCanvas.get_tk_widget().pack(fill="both", expand=True, padx=6, pady=6)
        self.fusionAxis.text(0.5, 0.5, "Waiting to Run", ha="center", va="center")
        self.fusionCanvas.draw()

    def BuildScatterFrame(self):
        self.scatterFigure = Figure(figsize=(8.2, 5.2), dpi=140)
        self.scatterAxis = self.scatterFigure.add_subplot(111)
        self.scatterCanvas = FigureCanvasTkAgg(self.scatterFigure, master=self.scatterFrame)
        self.scatterCanvas.get_tk_widget().pack(fill="both", expand=True, padx=6, pady=6)
        self.scatterAxis.text(0.5, 0.5, "Waiting to Run", ha="center", va="center")
        self.scatterCanvas.draw()

    def PostUiEvent(self, eventData):
        self.uiQueue.put(eventData)

    def ProcessUiQueue(self):
        while True:
            try:
                eventData = self.uiQueue.get_nowait()
            except queue.Empty:
                break

            eventType = eventData.get("type")
            if eventType == "log":
                self.AppendLog(eventData["message"])
            elif eventType == "initial_points":
                self.ShowInitialPoints(eventData["points"])
            elif eventType == "request_initial_csv":
                self.EnableInitialCsvInput(eventData["points"])
            elif eventType == "recommended_points":
                self.ShowRecommendedPoints(eventData["points"])
            elif eventType == "request_points":
                self.EnablePointInputs(eventData["points"])
            elif eventType == "input_error":
                self.AppendLog(f"Input validation failed: {eventData['message']}")
                messagebox.showerror("Input Error", eventData["message"])
            elif eventType == "state":
                self.latestState = eventData["state"]
                self.RedrawPlots()
            elif eventType == "finished":
                self.HandleRunFinished(eventData["result"])
            elif eventType == "error":
                self.HandleRunError(eventData["message"])

        self.root.after(100, self.ProcessUiQueue)

    def AppendLog(self, message):
        self.logText.insert("end", message + "\n")
        self.logText.see("end")

    def SelectFile(self, variable):
        filePath = filedialog.askopenfilename(
            title="Select CSV File",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            initialdir=self.baseDir,
        )
        if filePath:
            variable.set(filePath)

    def StartRun(self):
        lowCsvPath = self.lowCsvVar.get().strip()
        validationCsvPath = self.validationCsvVar.get().strip()
        bhfText = self.bhfVar.get().strip()

        if not lowCsvPath:
            messagebox.showerror("Input Error", "Select a low-fidelity data file first.")
            return
        if not os.path.exists(lowCsvPath):
            messagebox.showerror("Input Error", "The low-fidelity data file does not exist.")
            return

        try:
            bhfValue = int(bhfText)
        except Exception:
            messagebox.showerror("Input Error", "BHF must be an integer.")
            return

        if bhfValue < 6:
            messagebox.showerror("Input Error", "BHF must be at least 6.")
            return

        if validationCsvPath and not os.path.exists(validationCsvPath):
            messagebox.showerror("Input Error", "The validation data file does not exist.")
            return

        self.isRunning = True
        self.latestState = None
        self.currentPointCount = 0
        self.isAutoSubmittingPoints = False
        self.initialPointText.delete("1.0", "end")
        self.initialCsvVar.set("")
        for pointIndex in range(2):
            self.pointEntryVars[pointIndex].set("")
            self.pointStatuses[pointIndex] = None
            self.pointLabelVars[pointIndex].set(f"Point {pointIndex + 1}: waiting for recommendation")
            self.pointEntries[pointIndex].configure(state="disabled")
            self.pointSkipButtons[pointIndex].configure(state="disabled", text="Skip")
            self.pointConfirmButtons[pointIndex].configure(state="disabled", text="Confirm")
        self.pointSubmitButton.configure(state="disabled")
        self.initialBrowseButton.configure(state="disabled")
        self.initialSubmitButton.configure(state="disabled")
        self.startButton.configure(state="disabled")
        self.stopButton.configure(state="normal")
        self.logText.delete("1.0", "end")
        self.AppendLog("Starting the fusion workflow.")

        self.interactor = GuiInteractor(self)
        self.logHandler = TkLogHandler(self)
        logger.addHandler(self.logHandler)

        self.workerThread = threading.Thread(
            target=self.RunFusionWorker,
            args=(lowCsvPath, validationCsvPath if validationCsvPath else None, bhfValue),
            daemon=True,
        )
        self.workerThread.start()

    def RunFusionWorker(self, lowCsvPath, validationCsvPath, bhfValue):
        try:
            resultInfo = SequentialFusion(
                cfdCsv=lowCsvPath,
                bhf=bhfValue,
                maxIters=50,
                randomState=42,
                krgCorr=BEST_KRG_CORR,
                validationCsv=validationCsvPath,
                modelConfig=BEST_MODEL_CONFIG,
                outputBaseDir=self.baseDir,
                interactor=self.interactor,
            )
            self.PostUiEvent({"type": "finished", "result": resultInfo})
        except Exception:
            self.PostUiEvent({"type": "error", "message": traceback.format_exc()})

    def RequestStop(self):
        if self.interactor is not None:
            self.interactor.RequestStop()
            self.AppendLog("Sent early-termination request.")

    def ShowInitialPoints(self, points):
        self.initialPointText.delete("1.0", "end")
        self.initialPointText.insert("end", "Initial high-fidelity recommended points:\n")
        for pointIndex, point in enumerate(points):
            self.initialPointText.insert("end", f"Point {pointIndex + 1}: {np.asarray(point).tolist()}\n")
        self.initialPointText.see("end")

    def EnableInitialCsvInput(self, points):
        self.ShowInitialPoints(points)
        self.initialBrowseButton.configure(state="normal")
        self.initialSubmitButton.configure(state="normal")
        self.AppendLog("Please submit the initial high-fidelity CSV file.")

    def SubmitInitialCsv(self):
        filePath = self.initialCsvVar.get().strip()
        if not filePath:
            messagebox.showerror("Input Error", "Please select the initial high-fidelity CSV file.")
            return
        if not os.path.exists(filePath):
            messagebox.showerror("Input Error", "The initial high-fidelity CSV file does not exist.")
            return
        if self.interactor is not None:
            self.interactor.SubmitInitialHighCsv(filePath)
            self.initialBrowseButton.configure(state="disabled")
            self.initialSubmitButton.configure(state="disabled")
            self.AppendLog(f"Submitted initial high-fidelity CSV: {filePath}")

    def ShowRecommendedPoints(self, points):
        self.AppendLog("Received a new round of recommended points.")
        for pointIndex, point in enumerate(points):
            self.AppendLog(f"Recommended point {pointIndex + 1}: {np.asarray(point).tolist()}")

    def EnablePointInputs(self, points):
        self.currentPointCount = len(points)
        self.isAutoSubmittingPoints = False
        for pointIndex in range(2):
            self.pointStatuses[pointIndex] = None
            if pointIndex < len(points):
                self.pointLabelVars[pointIndex].set(f"Point {pointIndex + 1}: {np.asarray(points[pointIndex]).tolist()}")
                self.pointEntryVars[pointIndex].set("")
                self.pointEntries[pointIndex].configure(state="normal")
                self.pointSkipButtons[pointIndex].configure(state="normal", text="Skip")
                self.pointConfirmButtons[pointIndex].configure(state="normal", text="Confirm")
            else:
                self.pointLabelVars[pointIndex].set(f"Point {pointIndex + 1}: none")
                self.pointEntryVars[pointIndex].set("")
                self.pointEntries[pointIndex].configure(state="disabled")
                self.pointSkipButtons[pointIndex].configure(state="disabled", text="Skip")
                self.pointConfirmButtons[pointIndex].configure(state="disabled", text="Confirm")
        self.pointSubmitButton.configure(state="disabled")
        self.AppendLog("Enter high-fidelity values for the current recommended points, then confirm each one or use Skip.")

    def ToggleSkip(self, pointIndex):
        if pointIndex >= self.currentPointCount:
            return

        currentStatus = self.pointStatuses[pointIndex]
        if currentStatus == "skip":
            self.pointStatuses[pointIndex] = None
            self.pointEntryVars[pointIndex].set("")
            self.pointEntries[pointIndex].configure(state="normal")
            self.pointConfirmButtons[pointIndex].configure(state="normal", text="Confirm")
            self.pointSkipButtons[pointIndex].configure(text="Skip")
        else:
            self.pointStatuses[pointIndex] = "skip"
            self.pointEntryVars[pointIndex].set("skip")
            self.pointEntries[pointIndex].configure(state="disabled")
            self.pointConfirmButtons[pointIndex].configure(state="disabled", text="Confirm")
            self.pointSkipButtons[pointIndex].configure(text="Cancel Skip")

        self.UpdatePointSubmitState()

    def ConfirmPoint(self, pointIndex):
        if pointIndex >= self.currentPointCount:
            return

        currentStatus = self.pointStatuses[pointIndex]
        if currentStatus == "confirmed":
            self.pointStatuses[pointIndex] = None
            self.pointEntries[pointIndex].configure(state="normal")
            self.pointConfirmButtons[pointIndex].configure(text="Confirm")
            self.pointSkipButtons[pointIndex].configure(state="normal")
            self.AppendLog(f"Cancelled confirmation for recommended point {pointIndex + 1}.")
            self.UpdatePointSubmitState()
            return

        responseText = self.pointEntryVars[pointIndex].get().strip()
        if not responseText:
            messagebox.showerror("Input Error", f"Recommended point {pointIndex + 1} has no input.")
            return
        if responseText.lower() == "skip":
            messagebox.showerror("Input Error", f"Recommended point {pointIndex + 1} is currently skipped and does not need confirmation.")
            return

        self.pointStatuses[pointIndex] = "confirmed"
        self.pointEntries[pointIndex].configure(state="disabled")
        self.pointConfirmButtons[pointIndex].configure(text="Cancel Confirm")
        self.pointSkipButtons[pointIndex].configure(state="disabled")
        self.AppendLog(f"Confirmed input for recommended point {pointIndex + 1}.")
        self.UpdatePointSubmitState()

    def AutoSubmitPointInputs(self):
        if self.isAutoSubmittingPoints:
            return
        self.isAutoSubmittingPoints = True
        try:
            self.AppendLog("All recommended points in this round have been confirmed or skipped; submitting automatically.")
            self.SubmitPointInputs()
        finally:
            self.isAutoSubmittingPoints = False

    def UpdatePointSubmitState(self):
        readyToSubmit = True
        for pointIndex in range(self.currentPointCount):
            if self.pointStatuses[pointIndex] not in {"confirmed", "skip"}:
                readyToSubmit = False
                break

        self.pointSubmitButton.configure(state="normal" if readyToSubmit and self.currentPointCount > 0 else "disabled")
        if readyToSubmit and self.currentPointCount > 0 and self.interactor is not None and not self.isAutoSubmittingPoints:
            self.root.after(0, self.AutoSubmitPointInputs)

    def SubmitPointInputs(self):
        responseList = []
        for pointIndex in range(self.currentPointCount):
            if self.pointStatuses[pointIndex] not in {"confirmed", "skip"}:
                messagebox.showerror("Input Error", f"Recommended point {pointIndex + 1} has not been confirmed or skipped.")
                return
            responseText = self.pointEntryVars[pointIndex].get().strip()
            if not responseText:
                messagebox.showerror("Input Error", f"Please fill in recommended point {pointIndex + 1}, or click Skip.")
                return
            responseList.append(responseText)

        if self.interactor is not None:
            self.interactor.SubmitPointResponses(responseList)

        for pointIndex in range(2):
            self.pointStatuses[pointIndex] = None
            self.pointEntries[pointIndex].configure(state="disabled")
            self.pointSkipButtons[pointIndex].configure(state="disabled", text="Skip")
            self.pointConfirmButtons[pointIndex].configure(state="disabled", text="Confirm")
        self.pointSubmitButton.configure(state="disabled")
        self.AppendLog("Submitted current round recommended point inputs.")

    def RedrawPlots(self):
        self.DrawFusionFigure()
        self.DrawScatterFigure()

    def EnsureFusionAxis(self, dim):
        needs3d = dim == 2
        currentName = getattr(self.fusionAxis, "name", "")
        if needs3d and currentName != "3d":
            self.fusionFigure.clear()
            self.fusionAxis = self.fusionFigure.add_subplot(111, projection="3d")
        elif not needs3d and currentName == "3d":
            self.fusionFigure.clear()
            self.fusionAxis = self.fusionFigure.add_subplot(111)

    def DrawFusionFigure(self):
        if self.latestState is None:
            self.EnsureFusionAxis(dim=1)
            self.fusionAxis.clear()
            self.fusionAxis.text(0.5, 0.5, "Waiting to Run", ha="center", va="center")
            self.fusionCanvas.draw_idle()
            return

        stateData = self.latestState
        dim = stateData["dim"]
        self.EnsureFusionAxis(dim)
        self.fusionAxis.clear()

        if dim == 1:
            bounds = stateData["bounds"]
            xMin, xMax = bounds[0]
            xPlot = np.linspace(xMin, xMax, 300).reshape(-1, 1)

            if self.showFusionVar.get() and stateData["fusionModel"] is not None:
                meanFusion, stdFusion = PredictFusionRows(
                    fusionModel=stateData["fusionModel"],
                    xData=xPlot,
                    scalerXLow=stateData["scalerXLow"],
                    scalerYLow=stateData["scalerYLow"],
                    returnStd=True,
                )
                self.fusionAxis.plot(xPlot[:, 0], meanFusion, color="#2f7ed8", linewidth=2.0, label="Fusion model")
                if self.showConfidenceVar.get():
                    self.fusionAxis.fill_between(
                        xPlot[:, 0],
                        meanFusion - 1.96 * stdFusion,
                        meanFusion + 1.96 * stdFusion,
                        color="#2f7ed8",
                        alpha=0.16,
                        label="95% confidence interval",
                    )

            if self.showLowVar.get():
                self.fusionAxis.scatter(stateData["xLow"][:, 0], stateData["yLow"], color="#4f81bd", alpha=0.45, s=24, label="Low-fidelity samples")
            if self.showHighVar.get() and len(stateData["xHigh"]) > 0:
                self.fusionAxis.scatter(stateData["xHigh"][:, 0], stateData["yHigh"], color="#c0504d", s=36, label="High-fidelity samples")
            if self.showValidationVar.get() and stateData["xVal"] is not None and stateData["yVal"] is not None:
                sortIndex = np.argsort(stateData["xVal"].ravel())
                self.fusionAxis.plot(stateData["xVal"].ravel()[sortIndex], stateData["yVal"].ravel()[sortIndex], color="black", linewidth=1.3, label="Validation data")

            self.fusionAxis.set_xlabel(stateData["inputColumns"][0] if stateData["inputColumns"] else "Input variable")
            self.fusionAxis.set_ylabel(stateData["outputColumn"] or "Output variable")
            self.fusionAxis.set_title(f"Real-time Fusion Result (iteration {stateData['iterations']} / BHF {stateData['BHF']})")
            self.fusionAxis.grid(True, alpha=0.25)
            if self.fusionAxis.get_legend_handles_labels()[0]:
                self.fusionAxis.legend(loc="best")
            self.fusionCanvas.draw_idle()
            return

        if dim == 2:
            bounds = stateData["bounds"]
            x1Min, x1Max = bounds[0]
            x2Min, x2Max = bounds[1]
            x1Grid = np.linspace(x1Min, x1Max, 80)
            x2Grid = np.linspace(x2Min, x2Max, 80)
            meshX1, meshX2 = np.meshgrid(x1Grid, x2Grid)
            gridPoints = np.vstack([meshX1.ravel(), meshX2.ravel()]).T

            if self.showFusionVar.get() and stateData["fusionModel"] is not None:
                meanFusion = PredictFusionRows(
                    fusionModel=stateData["fusionModel"],
                    xData=gridPoints,
                    scalerXLow=stateData["scalerXLow"],
                    scalerYLow=stateData["scalerYLow"],
                    returnStd=False,
                )
                meshZ = meanFusion.reshape(meshX1.shape)
                self.fusionAxis.plot_surface(
                    meshX1,
                    meshX2,
                    meshZ,
                    cmap="viridis",
                    edgecolor="none",
                    alpha=0.82,
                    antialiased=True,
                )

            if self.showLowVar.get():
                self.fusionAxis.plot_trisurf(
                    stateData["xLow"][:, 0],
                    stateData["xLow"][:, 1],
                    stateData["yLow"],
                    cmap="coolwarm",
                    alpha=0.35,
                    edgecolor="none",
                    linewidth=0,
                    antialiased=True,
                )
            if self.showHighVar.get() and len(stateData["xHigh"]) > 0:
                self.fusionAxis.scatter(
                    stateData["xHigh"][:, 0],
                    stateData["xHigh"][:, 1],
                    stateData["yHigh"],
                    color="#c0504d",
                    marker="^",
                    s=36,
                    edgecolors="k",
                    depthshade=False,
                )
            if self.showValidationVar.get() and stateData["xVal"] is not None and stateData["yVal"] is not None:
                self.fusionAxis.plot_trisurf(
                    stateData["xVal"][:, 0],
                    stateData["xVal"][:, 1],
                    stateData["yVal"],
                    cmap="gray",
                    alpha=0.26,
                    edgecolor="none",
                    linewidth=0,
                    antialiased=True,
                )

            self.fusionAxis.set_xlabel(stateData["inputColumns"][0] if stateData["inputColumns"] else "Input variable 1")
            self.fusionAxis.set_ylabel(stateData["inputColumns"][1] if len(stateData["inputColumns"]) > 1 else "Input variable 2")
            self.fusionAxis.set_zlabel(stateData["outputColumn"] or "Output variable")
            self.fusionAxis.set_title(f"Real-time Fusion Result (iteration {stateData['iterations']} / BHF {stateData['BHF']})")
            self.fusionAxis.view_init(elev=28, azim=-58)
            self.fusionCanvas.draw_idle()
            return

        self.fusionAxis.text2D(0.5, 0.5, "The current input dimension exceeds 2; fusion visualization is not displayed.", ha="center", va="center", transform=self.fusionAxis.transAxes) if getattr(self.fusionAxis, "name", "") == "3d" else self.fusionAxis.text(0.5, 0.5, "The current input dimension exceeds 2; fusion visualization is not displayed.", ha="center", va="center")
        self.fusionCanvas.draw_idle()

    def DrawScatterFigure(self):
        self.scatterAxis.clear()

        if self.latestState is None or self.latestState["fusionModel"] is None:
            self.scatterAxis.text(0.5, 0.5, "Waiting for Model Training", ha="center", va="center")
            self.scatterCanvas.draw_idle()
            return

        stateData = self.latestState
        if stateData["xVal"] is not None and stateData["yVal"] is not None:
            xEval = stateData["xVal"]
            yTrue = stateData["yVal"]
        else:
            xEval = stateData["xHigh"]
            yTrue = stateData["yHigh"]

        if xEval is None or len(xEval) == 0:
            self.scatterAxis.text(0.5, 0.5, "No data available for regression fitting", ha="center", va="center")
            self.scatterCanvas.draw_idle()
            return

        yPred = PredictRows(
            fusionModel=stateData["fusionModel"],
            xData=xEval,
            scalerXLow=stateData["scalerXLow"],
            scalerYLow=stateData["scalerYLow"],
        )
        rmseValue, maeValue, r2Value = ComputeMetrics(yTrue, yPred)

        self.scatterAxis.scatter(yTrue, yPred, color="#1f77b4", alpha=0.75)
        minValue = float(min(np.min(yTrue), np.min(yPred)))
        maxValue = float(max(np.max(yTrue), np.max(yPred)))
        if abs(maxValue - minValue) < 1e-12:
            maxValue = minValue + 1.0
        self.scatterAxis.plot([minValue, maxValue], [minValue, maxValue], color="#d62728", linestyle="--")
        self.scatterAxis.set_xlabel("True physical value")
        self.scatterAxis.set_ylabel("Fusion model prediction")
        self.scatterAxis.set_title(f"RMSE={rmseValue:.6f}  MAE={maeValue:.6f}  R2={r2Value:.6f}")
        self.scatterAxis.grid(True, alpha=0.25)
        self.scatterCanvas.draw_idle()

    def HandleRunFinished(self, resultInfo):
        if self.logHandler is not None:
            logger.removeHandler(self.logHandler)
            self.logHandler = None

        self.isRunning = False
        self.startButton.configure(state="normal")
        self.stopButton.configure(state="disabled")
        self.isAutoSubmittingPoints = False
        self.initialBrowseButton.configure(state="disabled")
        self.initialSubmitButton.configure(state="disabled")
        self.pointSubmitButton.configure(state="disabled")
        for pointIndex in range(2):
            self.pointStatuses[pointIndex] = None
            self.pointEntries[pointIndex].configure(state="disabled")
            self.pointSkipButtons[pointIndex].configure(state="disabled", text="Skip")
            self.pointConfirmButtons[pointIndex].configure(state="disabled", text="Confirm")

        self.AppendLog("Fusion workflow completed.")
        self.AppendLog(f"Result directory: {resultInfo['resultDir']}")
        self.AppendLog(f"Model file: {resultInfo['modelPath']}")
        self.AppendLog(f"Prediction results: {resultInfo['predictionCsvPath']}")
        self.AppendLog(f"Regression-fit scatter plot: {resultInfo['scatterSvgPath']}")
        messagebox.showinfo("Run Complete", f"Results saved to:\n{resultInfo['resultDir']}")

    def HandleRunError(self, errorMessage):
        if self.logHandler is not None:
            logger.removeHandler(self.logHandler)
            self.logHandler = None

        self.isRunning = False
        self.startButton.configure(state="normal")
        self.stopButton.configure(state="disabled")
        self.isAutoSubmittingPoints = False
        self.AppendLog("Run failed.")
        self.AppendLog(errorMessage)
        messagebox.showerror("Run Failed", errorMessage)


def Main():
    root = tk.Tk()
    FusionGuiApp(root)
    root.mainloop()


if __name__ == "__main__":
    Main()
