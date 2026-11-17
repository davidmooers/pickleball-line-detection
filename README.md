# Vibration-Based Pickleball Impact Detection and Localization
CSC 475 - Project Design Specification  
David Mooers

## Project Overview
This project will investigate whether surface vibrations measured using piezoelectric sensors can be used to detect pickleball impacts,
distinguish them from other vibrations sources, and estimate where an impact occurred relative to a court line.

The work is divided into three phases:
1. Vibration Detection and Source Classification
2. Multi-Sensor Impact Localization
3. Simple Court Line Demo

## Research Questions
- Can a piezoelectric sensor reliably detect a pickleball bounce?
- Can vibration signals distinguish pickleball bounces from footsteps and other disturbances?
- Can multiple sensors approximate an impact location, and with what accuracy?
- How well does the combined detection, classification, and localization system perform on a real pickleball court?

## System Design

### Hardware
- Piezoelectric discs to measure surface vibrations.
- Arduino microcontroller to sample the sensor signals and stream data to a laptop.
- A circuit which buffers and amplifies the sensor's data signal.
- A laptop to record samples, process events, and display information.

### Circuit Diagram
TODO

### Data Flow
1. An impact produces vibrations in the surface.
2. The piezoelectric sensor converts those vibrations into an electrical signal.
3. The Arduino samples the signal and transmits the timestamped measurements.
4. A Python program reads data stream from Arduino to detect and record vibration events.
5. Features are extracted from each event and used to classify the vibration source.
6. When an event is classifed as a pickleball bounce, measurements from multiple sensors are used to estimate the position.

### Tools and Technology
- Arduino C/C++ for data acquisition.
- Python and pyserial for serial communication and recording.
- NumPy and SciPy for signal processing.
- scikit-learn for classification and position estimation.
- Matplotlib for waveforms, spectra, and evaluation plots.
- Git for version control.

## Dataset and Experimental Procedure
I will collect a labelled dataset using the piezoelectric sensors. Each recording will include raw samples, timestamps, event type, recording session, sensor configuration, and impact position (where applicable). Drop height, surface, and sensor distance will also be recorded when controlled.

The primary event labels will be <b>pickleball bounce</b> and <b>non-pickleball bounce</b>. Negative events will retain sublabels such as footsteps, paddle tap, distant games, etc. so I can determine which disturbances are hardest to reject. Quiet recordings will also be used to measure background noise and false triggers.

The initial collection target is approximately 200 examples per class across three sessions. These are proposed targets are subject to change. Repeated events will vary in position and impact strength so that the classifier is not under trained.

Recordings from different collection sessions will be used to train and test the classifier. Some training recordings will be set aside to compare models and adjust their settings. The final test recordings will only be used after these choices are complete.

I will record how many impacts I produce during each test and compare this with how many the system detects. Recordings with distorted signals, missing samples, or unclear labels will be marked.

## Phase 1 - Vibration Detection and Source Classification

### Goal
Detect individual vibration events and classify each event as either:
- Pickleball bounce
- Non-pickleball bounce

Non-pickleball bounces include footsteps, surface taps, and other disturbances. Paddle strikes will be tested as possible interference.

### Setup and Method

- Attach one piezoelectric disc to a quiet test surface using a consistent mounting method, initially electrical tape.
- Connect the sensor to the Arduino circuit to buffer and amplify data signal.
- Stream timestamped ADC samples to a Python program on my laptop.
- Measure background noise and the actual sampling rate.
- Mark reference distances approximately 10 cm, 30 cm, and 50 cm from the sensor, adjusting the range if necessary.
- Record and label ball drops at different distances and heights, footsteps, and other disturbances.
- Detect events using short-window energy or peak-to-peak amplitude. Capture samples before the trigger and the vibration response after it.
- Require a sustained quiet period before re-arming the detector, and evaluate whether ringing causes duplicate events.
- Extract features such as peak amplitude, RMS, energy, duration, decay, dominant frequency, and frequency-band energy.
- Compare an amplitude/energy threshold baseline with Random Forest and Support Vector Machine classifiers.

### Evaluation

Phase 1 will measure:
- Detection rate at different distances: detected bounces divided by attempted bounces.
- Signal strength relative to background noise.
- False triggers per minute and duplicate detections per impact.
- Classification accuracy, pickleball precision, recall, F1-score, and a confusion matrix on held-out detected events.
- The proportion of all attempted bounces both detected and correctly classified.
Separating detection from classification will make it clear whether failures come from missing the vibration or misidentifying a captured event.

## Phase 2 - Multi-Sensor Impact Localization

### Goal
Use multiple piezoelectric sensors to estimate where a pickleball bounce occurred within a small test area.

### Setup and Method
- Place at least three sensors at known positions around the test area.
- Mark several impact positions and record repeated ball drops at each position.
- Compare the signals recorded by the sensors, including their strength and timing.
- Investigate how these differences can be used to estimate the impact position.
- Test the method using new ball drops, including positions between the marked points.

The sampling rate and timing differences between sensor readings will be measured to determine whether arrival times can provide useful location information.

### Evaluation

Compare the estimated impact position with the actual position and measure the average distance error in centimetres.
Repeat drops at the same positions to check how consistent the estimates are. Examine whether accuracy changes across the test area, particularly as impacts move farther from the sensors.

This phase will use isolated pickleball bounces so that localization can be evaluated separately from source classification.

## Phase 3 - Simple Court Line Demo

### Goal
Combine detection, classification, and localization on a small section of a real pickleball court line.

### Setup and Method
- Place multiple sensors around a short line section, approximately 0.5–1 m long.
- Verify that court bounces are detectable and collect court-specific calibration data.
- Drop pickleballs at known locations inside, outside, and near the line.
- Detect whether each event was a pickleball bounce and estimate its location.
- Compare the estimated position with the measured boundary to produce an experimental IN/OUT decision.
- Test the system using at least 100 new pickleball bounces, 50 other impacts such as footsteps or taps, and ten minutes of background recording.

Tests will include clearly separated positions and positions progressively closer to the line, such as 20 cm, 10 cm, and 5 cm from its outer edge. Ground truth will use measured reference positions and video where available. Uncertain landing positions will be flagged rather than assigned precise labels without evidence.
Contact with the painted line will be labelled on the in side for this experiment. Because a point estimate does not capture the entire ball-surface contact footprint, very close cases may require an uncertain result rather than a forced decision.

### Evaluation
Measure:
- How many pickleball bounces the system correctly detects and how often it mistakes other vibrations for bounces.
- The average distance between the estimated and actual impact positions.
- How often the system makes the correct IN/OUT call, misses a bounce, or cannot make a decision.
- How accuracy changes as bounces land closer to the line.
- How often footsteps, other impacts, or background vibrations cause false detections.
- The time between a bounce and the system displaying its result.

These results will show how well the prototype works on a small court section and what would need to improve before testing a larger area.


## Individual Objectives

### David Mooers
I will be responsible for the circuit and acquisition setup, data collection, Python processing, model development, evaluation, and documentation.

### Objective 1: Detect vibrations and identify pickleball bounces

- PI1 (basic): Set up one sensor, measure background noise and how often readings are taken, and save the readings with their times.
- PI2 (basic): Build a program that records individual impacts. Test how often it misses an impact or counts the same impact more than once.
- PI3 (expected): Collect and label recordings across at least three sessions. Measure signal characteristics such as strength, duration, and frequency.
- PI4 (expected): Compare three methods for identifying pickleball bounces using separate training and test recordings. Measure how often each method correctly identifies bounces and other impacts.
- PI5 (advanced): Test different sensor distances, attachment methods, and bounces occurring alongside footsteps. Make and evaluate one improvement based on the results.

### Objective 2: Estimate impact position using multiple sensors

- PI1 (basic): Record signals from at least three sensors and measure how often each sensor is read and the time between their readings.
- PI2 (basic): Record repeated ball drops at marked positions. Test whether the sensor with the strongest signal is also the sensor closest to the impact.
- PI3 (expected): Develop a method that estimates impact position by comparing the strength and/or timing of signals from multiple sensors.
- PI4 (expected): Test the method with new recordings, including impacts between the marked positions. Measure how far estimates are from the actual positions, how consistent they are, and how errors vary across the area.
- PI5 (advanced): Compare another estimation method or sensor arrangement using the same test impacts, and identify where each approach works well or struggles.

### Objective 3: Demonstrate the combined system on a court-line section

- PI1 (basic): Place sensors around a short section of a court line and check whether they can detect nearby pickleball bounces.
- PI2 (basic): Combine bounce detection, bounce identification, position estimation, and the displayed IN/OUT result into one working program.
- PI3 (expected): Collect court recordings to set up the system, then use new recordings to test bounces inside, outside, and near the line, along with other impacts.
- PI4 (expected): Measure how often the system detects bounces, how far its position estimates are from the actual landing positions, and how often it makes correct IN/OUT calls. Also record missed bounces, cases without a decision, and response time.
- PI5 (advanced): Test bounces near footsteps or occurring at the same time. Allow the system to report "uncertain" and measure whether this reduces incorrect calls and how often it prevents a decision.

## Current Progress

A basic single-sensor system has been implemented to record vibrations and classify impacts. Initial trials suggest that it can distinguish pickleball bounces from other impacts, but formal testing has not yet been completed. Further recordings and evaluation are needed to measure its accuracy and consistency.

## Proposed Timeline

| Week | Planned work | Deliverable |
| ---- | ------------ | ----------- |
| 1 | Review the existing single-sensor system. Document the circuit and test detection at different distances. | Circuit diagram and initial detection results |
| 2 | Collect and label more recordings. Formally test and compare bounce classification methods. | Labelled dataset and classification results |
| 3 | Set up recording from at least three sensors. Compare their responses to impacts at known positions. | Working multi-sensor setup and sample recordings |
| 4 | Develop and test a method for estimating impact position. | Position estimation method and measured location errors |
| 5 | Combine classification and position estimation on a small court-line section. Test bounces inside, outside, and near the line. | Court-line demonstration and evaluation results |
| 6 | Analyse the results and limitations. Complete the report, instructions, and presentation. | Final report, instructions, and presentation |

### Deliverables and Limitations
Deliverables will include acquisition and processing code, labelled recordings with metadata, classification comparisons, localization results, and a final report. The court demonstration depends on usable court sensitivity and localization performance.

The main risks are weak sensor coupling, attachment differences, clipped signals, limited sampling resolution, reflections, and overlapping impacts. Pilot experiments will measure these issues before adding complexity. If court signals are too weak, localization will still be evaluated on the controlled surface, and the court limitation will be documented.

All dataset sizes, experiments, and performance indicators above are proposed work. Accuracy and precision will be reported from measurements rather than assumed in advance.

## Related Work

### Vibration-Based Localization

Poston, Buehrer, and Tarazaga [1] investigate indoor footstep localization using structural vibration instrumentation and address difficulties with arrival-time estimation and sensor geometry. Mirshekari et al. [2] examine localization in heterogeneous floors, including dispersion-related distortion. These studies motivate measuring propagation on the actual test surface rather than assuming ideal travel times.

Bahroun et al. [3] propose a localization method based on the signs of arrival-time differences in damped, dispersive floors. Alajlouni and Tarazaga [4] investigate energy-based localization with lower sampling requirements than arrival-time approaches. Together they motivate comparing timing features with an energy-based alternative when acquisition bandwidth is limited.

### Piezoelectric Ball and Surface Impact Sensing

Blank, Kautz, and Eskofier [5] use three piezoelectric sensors to localize ball impacts on table tennis rackets. Their calibration-based approach is relevant to this project's small sensor array, but performance on a racket does not establish achievable accuracy on a pickleball court.

Kundu et al. [6] localize impacts on anisotropic composite plates using passive sensor measurements. Perelli et al. [7] address dispersion in guided-wave impact localization through frequency-warped cross-wavelet analysis. These works motivate checking whether propagation changes signal shape enough to make simple correlation unreliable. Their advanced compensation methods are background rather than required implementations.

### Signal Processing and Learning Methods

Knapp and Carter [8] provide the generalized correlation framework for delay estimation, informing the proposed comparison of correlation and onset timing. Welch [9] motivates averaged spectral estimates; Harris [10] discusses spectral window selection, and Cooley and Tukey [11] provide the FFT foundation for frequency analysis.

Breiman's Random Forests [12] and Cortes and Vapnik's support-vector networks [13] motivate the two proposed learned classifiers. They will be compared using the same handcrafted vibration features rather than assuming that either model will perform best.

The scikit-learn [14] and SciPy [15] papers describe the software foundations used for modelling and signal analysis. These methodological references support implementation choices; they do not provide evidence that the proposed system can achieve court-line precision.

## References

[1] J. D. Poston, R. M. Buehrer, and P. A. Tarazaga, “Indoor footstep localization from structural dynamics instrumentation,” *Mechanical Systems and Signal Processing*, 2017. https://doi.org/10.1016/j.ymssp.2016.11.023

[2] M. Mirshekari, S. Pan, J. Fagert, E. M. Schooler, P. Zhang, and H. Y. Noh, “Occupant localization using footstep-induced structural vibration,” *Mechanical Systems and Signal Processing*, 2018. https://doi.org/10.1016/j.ymssp.2018.04.026

[3] R. Bahroun, O. Michel, F. Frassati, M. Carmona, and J. L. Lacoume, “New algorithm for footstep localization using seismic sensors in an indoor environment,” *Journal of Sound and Vibration*, vol. 333, no. 3, pp. 1046–1066, 2014. https://doi.org/10.1016/j.jsv.2013.10.004

[4] S. Alajlouni and P. Tarazaga, “A passive energy-based method for footstep impact localization, using an underfloor accelerometer sensor network with Kalman filtering,” *Journal of Vibration and Control*, vol. 26, nos. 11–12, 2020. https://doi.org/10.1177/1077546319890520

[5] P. Blank, T. Kautz, and B. Eskofier, “Ball impact localization on table tennis rackets using piezo-electric sensors,” *Proceedings of the 2016 ACM International Symposium on Wearable Computers*, pp. 72–79, 2016. https://doi.org/10.1145/2971763.2971778

[6] T. Kundu, S. Das, S. A. Martin, and K. V. Jata, “Locating point of impact in anisotropic fiber reinforced composite plates,” *Ultrasonics*, vol. 48, no. 3, pp. 193–201, 2008. https://doi.org/10.1016/j.ultras.2007.12.001

[7] A. Perelli, L. De Marchi, A. Marzani, and N. Speciale, “Frequency warped cross-wavelet multiresolution analysis of guided waves for impact localization,” *Signal Processing*, vol. 96, part A, pp. 51–62, 2014. https://doi.org/10.1016/j.sigpro.2013.05.008

[8] C. H. Knapp and G. C. Carter, “The generalized correlation method for estimation of time delay,” *IEEE Transactions on Acoustics, Speech, and Signal Processing*, vol. 24, no. 4, pp. 320–327, 1976. https://doi.org/10.1109/TASSP.1976.1162830

[9] P. D. Welch, “The use of fast Fourier transform for the estimation of power spectra: A method based on time averaging over short, modified periodograms,” *IEEE Transactions on Audio and Electroacoustics*, vol. 15, no. 2, pp. 70–73, 1967. https://doi.org/10.1109/TAU.1967.1161901

[10] F. J. Harris, “On the use of windows for harmonic analysis with the discrete Fourier transform,” *Proceedings of the IEEE*, vol. 66, no. 1, pp. 51–83, 1978. https://doi.org/10.1109/PROC.1978.10837

[11] J. W. Cooley and J. W. Tukey, “An algorithm for the machine calculation of complex Fourier series,” *Mathematics of Computation*, vol. 19, no. 90, pp. 297–301, 1965. https://doi.org/10.1090/S0025-5718-1965-0178586-1

[12] L. Breiman, “Random Forests,” *Machine Learning*, vol. 45, pp. 5–32, 2001. https://doi.org/10.1023/A:1010933404324

[13] C. Cortes and V. Vapnik, “Support-vector networks,” *Machine Learning*, vol. 20, pp. 273–297, 1995. https://doi.org/10.1007/BF00994018

[14] F. Pedregosa et al., “Scikit-learn: Machine Learning in Python,” *Journal of Machine Learning Research*, vol. 12, pp. 2825–2830, 2011. https://jmlr.org/papers/v12/pedregosa11a.html

[15] P. Virtanen et al., “SciPy 1.0: fundamental algorithms for scientific computing in Python,” *Nature Methods*, vol. 17, pp. 261–272, 2020. https://doi.org/10.1038/s41592-019-0686-2