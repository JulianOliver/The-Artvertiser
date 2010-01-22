
#pragma once

#include <pthread.h>
#include <stdio.h>
#include "FProfiler/FThread.h"
#include "FProfiler/FSemaphore.h"
#include "FProfiler/FTime.h"
#include <cv.h>
#include <highgui.h>
#include <map>

class MultiThreadCapture;


class MultiThreadCaptureManager
{
public:
    MultiThreadCaptureManager() {};
    ~MultiThreadCaptureManager() { instance = NULL; }
    static MultiThreadCaptureManager* getInstance() { if ( instance == NULL ) instance = new MultiThreadCaptureManager(); return instance; }

    /// add the given MultiThreadCapture* for the given cam.
    void addCaptureForCam( CvCapture* cam, MultiThreadCapture* cap );
    /// fetch the MultiThreadCapture for the given cam. return NULL if not found.
    MultiThreadCapture* getCaptureForCam( CvCapture* cam );
    /// remove the MultiThreadCapture for the given cam.
    void removeCaptureForCam( CvCapture* cam );

private:

    typedef std::map<CvCapture* , MultiThreadCapture* > CaptureMap;
    CaptureMap capture_map;

    static MultiThreadCaptureManager* instance;
    FSemaphore lock;

};


class pyr_yape;
class keypoint;
class object_view;

class MultiThreadCapture : public FThread
{
public:
    /// caller retains ownership of the capture object
    MultiThreadCapture( CvCapture* capture );
    ~MultiThreadCapture();

    /// also process the image after capturing, into an image of the given
    /// width/height/bitdepth. if width is 0, don't process. retrieve with
    /// getLastProcessedFrame
    void setupResolution( int _width, int _height, int _nChannels )
        { width = _width, height = _height; nChannels = _nChannels; }

    /// set thu number of pyramid levels for the object_view object
    void setNumPyramidLevels( int nLevels );

    /// Put a pointer to the last processed (grayscale) frame for the detect thread in
    /// grayFrame, and its last raw version in rawFrame; also put a
    /// timestamp for when the frame was grabbed into timestamp.
    ///
    /// Similar for keypoints, num_keypoints, and object_input_view.
    ///
    /// If a new frame is not available, return false, unless
    /// block_until_available is true, in which case block until a new frame is
    /// available (or timeout and return false after 10s).
    ///
    /// You may pass NULL for grayFrame, rawFrame, or timestamp, and they
    /// will be ignored.
    ///
    /// The MultiThreadCapture instance keeps ownership of the returned
    /// pointers - they remain valid until the next call to getLastDetectFrame.
    /// Return true on success.
    bool getLastDetectFrame( IplImage** grayFrame, IplImage** rawFrame, FTime* timestamp_copy,
                               keypoint** keypoints=NULL, int *num_keypoints=NULL, object_view** object_input_view=NULL,
                               bool block_until_available = false );

    /// like getLastProcessedFrame but returns the last raw frame, rather than
    /// the last processed frame; maintains a separate available state for
    /// determining whether a frame is new or not.
    ///
    /// @see getLastDetectFrame
    bool getLastDrawFrame( IplImage** rawFrame, FTime* timestamp_copy,
                         keypoint** keypoints = NULL, int *num_keypoints=NULL,
                         bool block_until_available = false );

    /// Put a copy of the last frame captured into *last_frame_copy, and its timestamp
    /// into timestamp_copy (if non-NULL).
    ///
    /// If *last_frame_copy is NULL, a new IplImage will be constructed for you.
    /// If *last_frame_copy is not NULL, we check that the size, depth and chennels
    /// match. If they do, we copy frame data in and return true. If they don't, we
    /// return false.
    ///
    /// In any case, caller takes or retains ownership of *last_frame_copy.
    ///
    /// If a new frame is not available, return false, unless block_until_available
    /// is true, in which case block until a new frame is available (or timeout and
    /// return false after 10s).
    bool getCopyOfLastFrame( IplImage** last_frame_copy, FTime* timestamp_copy=NULL,
                            bool block_until_available = false );

    /// fetch last processed frame's features
    bool getLastProcessedFrameFeatures( keypoint* feature_storage, int feature_storage_size );

    /// get a pointer to the pointer to the keypoint detector
    /// must call releaseKeypointDetectorLock when done
    pyr_yape** getKeypointDetectorPtrPtr() { feature_detector_lock.Wait(); return &feature_detector; }
    void releaseKeypointDetectorLock() { feature_detector_lock.Signal(); }

    /// start capture
    void startCapture();

    /// stop capture
    void stopCapture();

private:

    /// overridden from base
    void ThreadedFunction();

	/// process thread
	void startProcessThread();
	void stopProcessThread();
	/// for pthreads interface
	static void* processPthreadFunc( void* );
	/// the work actually happens here
	void processThreadedFunction();

    /// swap detect thread pointers
    void swapDetectPointers();
    void swapDrawPointers();

    CvCapture* capture;

	// for the capture threa
    FSemaphore capture_frame_lock;
    IplImage* capture_frame;

	// for the process thread
	bool process_thread_should_exit;
	pthread_t process_thread;
	FSemaphore process_thread_semaphore;

    // lock for the last frame
    FSemaphore last_frame_lock;
    // double buffered
    // How this works: when requested via getLastProcessedFrame, we return processed (& last_frame + timestamp)
    // and swap processed and processed_ret pointers (same for last_frame + timestamp).
    // This way, during capture the capture thread is always writing to processed, and the
    // external requester only has processed_ret.
    // We are assuming that the external requester operates single-threadedly.
    IplImage *last_frame, *last_frame_ret, *last_frame_draw, *last_frame_draw_ret, *last_frame_working;
    IplImage *processed, *processed_ret, *processed_working;
    IplImage *frame_processsize;
    FTime* timestamp, *timestamp_ret, *timestamp_draw, *timestamp_draw_ret, *timestamp_working;


    // keypoint stuff

    FSemaphore feature_detector_lock;

    pyr_yape* feature_detector;
    // sorry, this is very ugly.
    // the feature_detector is constructed externally, when the classifier is loaded (see planar_object_recognizer.load())
    // access is via MultiThreadCapture::getKeypointDetectorPtrPtr() called through MultiGrab::Cam::getKeypointDetectorPtrPtr()
    //
	// sorry about how ugly this is. see MultiThreadCapture for the origin, planar_object_recognizer for the destination
	// (also search for 'new pyr_yape' in planar_object_recognizer)


    // double buffered: 2x for draw thread; 2x for detection thread
    // see last_frame description above for explanation
    keypoint *kp_working, *kp_draw, *kp_draw_ret, *kp_detect, *kp_detect_ret;
    int kp_working_count, kp_draw_count, kp_draw_ret_count, kp_detect_count, kp_detect_ret_count;
    // double buffered, as above
    object_view *object_input_view_working, *object_input_view, *object_input_view_ret;

    bool new_draw_frame_available, new_detect_frame_available;

    int width, height, nChannels, num_pyramid_levels;

};


