/***
 *  $Id: time_check.hpp 38 2003-10-28 14:20:23Z zola $
 **
 *  File: time_check.hpp
 *  Developed: Mar 25, 2003
 *
 *  Institute of Computer & Information Sciences
 *  Czestochowa University of Technology
 *  Dabrowskiego 73
 *  42-200 Czestochowa, Poland
 *  tel/fax: +48 (0....)34 3250589
 *
 *  Author: Jaroslaw Zola <zola@icis.pcz.pl>
 *  Copyright (c) 2003 Jaroslaw Zola <zola@icis.pcz.pl>
 *  For copyright details please contact the author.
 */

#ifndef TIME_CHECK_HPP
#define TIME_CHECK_HPP

#include <ctime>
#include <string>
#include <vector>
#include <sys/time.h>


/** Tool for program execution checkpointing.
 */
class time_check {
public:
    typedef unsigned size_type;

    /** Constructs time_check.
     *  @param name name of the object.
     *  @param n maximal number of check points.
     */
    explicit time_check(const char* name = "", size_type n = 128)
	: name_(name), hbeg_(false), hfin_(false), pos_(0),
	  tpts_(n, 0.0), ipts_(n, "") {}


    /** @return number of stored checkpoints.
     */
    size_type size() const { return pos_; }

    /** @return maximal number of checkpoints.
     */
    size_type capacity() const { return tpts_.size(); }


    /** Starts checkpointing.
     */
    void start() {
	hbeg_ = true;
	gettimeofday(&beg_, 0);
    } // start

    /** Stops checkpointing.
     */
    void stop() {
	hfin_ = true;
	gettimeofday(&fin_, 0);
    } // stop


    /** Creates checkpoint.
     *  @param id name of the checkpoint.
     */
    void check(const std::string& id = "") {
	if ((hbeg_) && (pos_ < tpts_.size())) {
	    tpts_[pos_] = get_time();
	    ipts_[pos_] = id;
	    ++pos_;
	}
    } // check


    /** @return time from start() to stop().
     */
    double total() const {
	return (hbeg_ && hfin_) ?
	    (tv2double(fin_) - tv2double(beg_)) : 0.0;
    } // total


    /** @return time between points n and n-1.
     */
    double at_point(size_type n) const {
	if (!((hbeg_) && (n < pos_))) return 0.0;
	return (n < 1) ? (tpts_[0] - tv2double(beg_))
	    : (tpts_[n] - tpts_[n-1]);
    } // at_point


    /** @return time to point n.
     */
    double to_point(size_type n) const {
	return ((hbeg_) && (n < pos_)) ? tpts_[n] : 0.0;
    } // to_point


    /** @return id of the n'th point
     */
    std::string point_id(size_type n) const {
	return ((hbeg_) && (n < pos_)) ? ipts_[n] : "";
    } // point_id


    friend std::ostream& operator<<(std::ostream&, const time_check&);

    double get_time() const {
	timeval t;
	gettimeofday(&t, 0);
	return tv2double(t);
    } // get_time


private:
    double tv2double(const timeval& t) const {
	return t.tv_sec + (0.000001 * t.tv_usec);
    } // tv2double


    std::string name_;

    bool hbeg_; // true if start() called
    timeval beg_;

    bool hfin_; // true if stop() called
    timeval fin_;

    size_type pos_; // number of checkpoints

    std::vector<double> tpts_; // values in the checkpoints
    std::vector<std::string> ipts_; // checkpoints ids

}; // class time_check



inline std::ostream& operator<<(std::ostream& out, const time_check& t) {
    out << t.name_ << " time check:\n";

    std::size_t i = 0;
    for (; i < t.pos_; ++i) {
	out << t.ipts_[i] << ": +" << t.at_point(i) << "s\n";
    }

    out << "----------------\n";

    std::size_t tt = static_cast<std::size_t>(t.total());
    std::size_t ht = tt / 3600;
    std::size_t mt = (tt % 3600) / 60;
    std::size_t st = (tt % 3600) % 60;

    out << "total: " << t.total() << "s\n";
    if ((ht > 0) || (mt > 0)) {
	out << "       " << ht << "h:" << mt << "m:" << st << "s\n";
    }

    return out;
} // operator <<

#endif // TIME_CHECK_HPP
