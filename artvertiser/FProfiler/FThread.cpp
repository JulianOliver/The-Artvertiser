/*
 *  FThread.h
 *  F
 *
 *  Created by damian on 25/5/08.
 *  Copyright 2008 frey damian@frey.co.nz. All rights reserved.
 *
 */

#include "FThread.h"
#include <pthread.h>
#include <sys/errno.h>
#include <signal.h>

void FThread::StartThread( int thread_priority )
{
	if ( thread_running ) {
	    printf("FThread::Start(): FThread %x already running\n", this );
        return;
	}
	thread_should_stop = false;

    pthread_attr_t thread_attr;
    pthread_attr_init(&thread_attr);
    pthread_attr_setdetachstate(&thread_attr, PTHREAD_CREATE_JOINABLE);
    // launch
	int result = pthread_create( &the_thread, &thread_attr, run_function, this );

    // priority
    if ( thread_priority > 0 )
    {
        printf("attempting to set thread priority to %i\n" ,thread_priority );
        struct sched_param param;
        param.sched_priority = thread_priority;
        int res = pthread_setschedparam( the_thread, SCHED_RR, &param );
        if ( res != 0 )
        {
            printf("pthread_setschedparam failed: %s\n",
                   (res == ENOSYS) ? "ENOSYS" :
                   (res == EINVAL) ? "EINVAL" :
                   (res == ENOTSUP) ? "ENOTSUP" :
                   (res == EPERM) ? "EPERM" :
                   (res == ESRCH) ? "ESRCH" :
                   "???"
                   );
        }
        printf("setsched -> %i\n", param.sched_priority );
    }

    int policy;
    struct sched_param param;
    pthread_getschedparam( the_thread, &policy, &param );
    printf("created FThread policy %s priority %i\n",
                   (policy == SCHED_FIFO)  ? "SCHED_FIFO" :
                   (policy == SCHED_RR)    ? "SCHED_RR" :
                   (policy == SCHED_OTHER) ? "SCHED_OTHER" :
                   "???",
                   param.sched_priority);


    pthread_attr_destroy( &thread_attr );
	assert( result == 0 );
	thread_running = true;
}

void FThread::StopThread()
{
	if ( !thread_running ) {
	    printf("FThread::Stop(): FThread %x not running\n", this );
        return;
	}
	printf("stopping FThread %x\n", this );
	thread_should_stop = true;
	void * ret;
	pthread_join( the_thread, &ret );
	thread_running = false;
}

/*
long FThread::GetCurrentThreadId()
{
	pthread_t id = pthread_self();
	return (long)id.p;
}*/
