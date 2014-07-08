#ifndef MACROS_HPP
#define MACROS_HPP

/* This file provides macros to help neaten up code and encapsulate various
 * preprocessor bits and pieces, particularly those relating to the number of
 * dimensions and colours in the simulation.
 */

#if NDIM==1
#define COORD_INDEX_ARGS(n) const int n ## 0
#endif
#if NDIM==2
#define COORD_INDEX_ARGS(n) const int n ## 0, const int n ## 1
#endif
#if NDIM==3
#define COORD_INDEX_ARGS(n) const int n ## 0, const int n ## 1, const int n ## 2
#endif
#if NDIM==4
#define COORD_INDEX_ARGS(n) const int n ## 0, const int n ## 1, \
    const int n ## 2, const int n ## 3
#endif
#if NDIM==5
#define COORD_INDEX_ARGS(n) const int n ## 0, const int n ## 1, \
    const int n ## 2, const int n ## 3, const int n ## 4
#endif
#if NDIM==6
#define COORD_INDEX_ARGS(n) const int n ## 0, const int n ## 1, \
    const int n ## 2, const int n ## 3, const int n ## 4, const int n ## 5
#endif
#if NDIM==7
#define COORD_INDEX_ARGS(n) const int n ## 0, const int n ## 1, \
    const int n ## 2, const int n ## 3, const int n ## 4, const int n ## 5, \
    const int n ## 6
#endif
#if NDIM==8
#define COORD_INDEX_ARGS(n) const int n ## 0, const int n ## 1, \
    const int n ## 2, const int n ## 3, const int n ## 4, const int n ## 5, \
    const int n ## 6, const int n ## 7
#endif
#if NDIM==9
#define COORD_INDEX_ARGS(n) const int n ## 0, const int n ## 1, \
    const int n ## 2, const int n ## 3, const int n ## 4, const int n ## 5, \
    const int n ## 6, const int n ## 7, const int n ## 8
#endif
#if NDIM==10
#define COORD_INDEX_ARGS(n) const int n ## 0, const int n ## 1, \
    const int n ## 2, const int n ## 3, const int n ## 4, const int n ## 5, \
    const int n ## 6, const int n ## 7, const int n ## 8, const int n ## 9
#endif

#endif
