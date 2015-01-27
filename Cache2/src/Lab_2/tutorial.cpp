/*
// File: tutorial.cpp
//              
// Tutorial implementation for Advances in Computer Architecture Lab session
// Implements a simple CPU and memory simulation with randomly generated
// read and write requests
//
// Author(s): Michiel W. van Tol, Mike Lankamp, Jony Zhang, 
//            Konstantinos Bousias
// Copyright (C) 2005-2009 by Computer Systems Architecture group, 
//                            University of Amsterdam
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.
//
*/

#include <systemc.h>
#include <aca2009.h>
#include <fstream>

using namespace std;

#define READ_MODE 0
#define WRITE_MODE 1

#define DEBUG 1
#define DEBUG_CPU 0
#define DEBUG_CACHE 0
#define DEBUG_BUS 1

static const int MEM_SIZE = 32768;   // 32kB = 32 x 1024 Byte
static const int LINE_SIZE = 32;     // 32B Line size
static const int ASSOCIATIVITY = 8;  // 8 way set assosiative
static const int N_SET = MEM_SIZE/(LINE_SIZE*ASSOCIATIVITY); // 128 Lines

// Log file for the results
ofstream Log;

// Cache line should have  
// Tag, Status bit and Data
typedef struct {
    int data[8];
    unsigned int status :1; /* Valid / Invalid */
    unsigned int tag : 20;     
}l1_cache_way;

// We need LRU for each set
typedef struct {
    l1_cache_way way[ASSOCIATIVITY];
    unsigned int lru : 7;
}l1_cache_set;

int pending_processors, desired_addresses;
class Cache;

enum Req {
    INVALID,
    READ,
    WRITE,
    READX
};

/* Bus interface, modified version from assignment. */
class Bus_if : public virtual sc_interface 
{
    public:
        virtual bool read(int writer, int addr) = 0;
        virtual bool write(int writer, int addr, int data) = 0;
        virtual bool readX(int writer, int addr) = 0;
        virtual int check_ongoing_requests(int writer, int addr, Req opr) = 0;
        virtual void release_mutex(int writer, int addr ) = 0;
};

/* Cache module, simulates the cache. */
SC_MODULE(Cache) 
{

public:
    /* Function type. */
    enum Function 
    {
        F_INVALID,
        F_READ,
        F_WRITE,
        F_READX
    };

    /* Return code to CPU. */
    enum RetCode 
    {
        RET_READ_DONE,
        RET_WRITE_DONE,
    };

    /* Possible line states depending on the cache coherence protocol. */
    enum Line_State 
    {
        INVALID,
        VALID
    };

    /* In/Out ports with the cpu, as you did in assignment 1 */
    sc_in<bool>     Port_CLK;
    sc_in<Function> Port_Func;
    sc_in<int>      Port_Addr;
    sc_out<RetCode> Port_Done;
    sc_inout_rv<32> Port_Data;

    /* Bus snooping ports. */
    sc_in_rv<32>    Port_BusAddr;
    sc_in<int>      Port_BusWriter;
    sc_in<Function> Port_BusValid;

    /* Bus requests ports. */
    sc_port<Bus_if> Port_Bus;

    /* Variables. */
    int cache_id;
    bool snooping;

    int cache_lookup(int address, int mode);
    int lru_line(int set_no);
    void update_lru_state(int set_no, int j);
    void invalidate_my_copy(int address);
    

    /* Constructor. */
    SC_CTOR(Cache) 
    {
        /* Create threads for handling data and snooping the bus. */
        // this is very important, and is the heart of assignment 2: to have two independent threads, one to snoop the bus, and the other to handle data.
        // your code should be mainly to implement these two threads.
        SC_THREAD(bus);
        SC_THREAD(execute);

        /* Listen to clock.*/
        sensitive << Port_CLK.pos();
        dont_initialize();  // decide to use this functionality or not for your threads. Refer to SC_THREAD help.
    
        // create and initialize your variables/private members here
        cache_set  = new l1_cache_set[N_SET];

        // Set tag = 1
        for (int i = 0; i < N_SET; ++i)
        {
            for (int j = 0; j < ASSOCIATIVITY; ++j)
            {
                cache_set[i].way[j].tag = 0xfffff;   // init all tag bits to 1
            }
        }

    }

    /* destructor. */    
    ~Cache() {
        delete[] cache_set;
    }
private:
    // define your cache structure here, and add a state to each line indicating the cache-coherence protocol
    l1_cache_set *cache_set;
    int data;

    /* Thread that handles the bus. */
    void bus() 
    {
        /* Continue while snooping is activated. */
        while(snooping)
        {
           wait(Port_BusValid.value_changed_event());

            int snoop_addr = Port_BusAddr.read().to_int();
            /* Possibilities. */
            switch(Port_BusValid.read())
            {
                // your code of what to do while snooping the bus
                // keep in mind that a certain cache should distinguish between bus requests made by itself and requests made by other caches.
                // count and report the total number of read/write operations on the bus, 
                // in which the desired address (by other caches) is found in the snooping cache (probe read hits and probe write hits).
                case F_INVALID: // Invalid request
                    break;
                case F_READ: // BusRd request 
                    if(Port_BusWriter.read() == cache_id) {
                        // own cache request
                        Log << "\t@" << sc_time_stamp() << ": Cache " <<cache_id << " snoops own read req \n";
                    }
                    else {
                        // Bus read : nothing to do in V,I protocol
                    }
                    break;
                case F_WRITE: // Bus wrire request
                    if(Port_BusWriter.read() == cache_id) {
                        // own cache request
                        Log << "\t@" << sc_time_stamp() << ": Cache " <<cache_id << " snoops own write req \n";
                    }
                    else {
                        // Bus write from other cache
                        invalidate_my_copy(snoop_addr);
                    }
                    break;
                case F_READX:
                    if(Port_BusWriter.read() == cache_id) {
                        // own cache request
                        Log << "\t@" << sc_time_stamp() << ": Cache " <<cache_id << " snoops own readx req \n";
                    }
                    else {
                        // Bus write from other cache
                        invalidate_my_copy(snoop_addr);
                    }
                    break;
                default:
                    Log << " @" << sc_time_stamp() << ": ------------ Error in snooping bus request ---------\n";
                    break;
            }
        }
    }

    /* Thread that handles the data requests. */
    void execute() 
    {
        //* Begin loop. */
        while(true)
        {
            /* wait for something to do... */
            wait(Port_Func.value_changed_event());

            /* Read or write? */
            int f = Port_Func.read();

            /* Read address. */
            int addr = Port_Addr.read();
            /* Calculate block and set, can also be done using bitshifts */
            // code that implements the cache, and send bus requests.
            data   = 0;
            int cache_data;

            if (f == F_READ) 
            {
                cache_data = cache_lookup(addr, READ_MODE);
                Port_Data.write(cache_data);
                if(DEBUG_CACHE) Log << "\t@" << sc_time_stamp() << ": Cache " << cache_id << " writing Port_Done \n";
                Port_Done.write( RET_READ_DONE );
                wait();
                Port_Data.write("ZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZ");
            }
            else if(f == F_WRITE)
            {
                data = Port_Data.read().to_int();
                cache_data = cache_lookup(addr, WRITE_MODE);
                if(DEBUG_CACHE) Log << "\t@" << sc_time_stamp() << ": Cache " << cache_id << " writing Port_Done \n";
                Port_Done.write( RET_WRITE_DONE );
                wait();
            }
            else {
                Port_Done.write( RET_WRITE_DONE );
                wait();
            }
        }
    }
};


// Member function definitions
int Cache::cache_lookup(int address, int mode)
{
    int set_no = (address & 4064) >> 5 ;      // 0b111111100000 = 4064
    unsigned int tag_no  = (address & 4294963200U) >> 12; // 0xfffff000 = 4294963200
    int word_in_line = (address & 28) >> 2;   // 0b11100 = 28
    int intended_data = 0, locked_mutex;
    bool found = false;
        
    if(DEBUG_CACHE) Log << "\t@" << sc_time_stamp() << ": cache lookup. Line no: " << set_no << endl;
    if (mode == 0) 
    {
        for (int i = 0; i < ASSOCIATIVITY; ++i)
        {
            if( (cache_set[set_no].way[i].status == 1) && (found == false) ){
                
                if (cache_set[set_no].way[i].tag == tag_no) 
                {
                    Log << "\t@" << sc_time_stamp() << ": C" << cache_id << " Read hit \n";
                    intended_data = cache_set[set_no].way[i].data[word_in_line];
                    stats_readhit(cache_id);
                    update_lru_state(set_no, i);
                    found = true;
                    wait();
                    break;
                }
            }
        }
        if (found == false)
        {
            if(DEBUG) Log << "\t@" << sc_time_stamp() << ": C" << cache_id << " Read miss: Tag: " << tag_no << endl;
            // Read miss. Bring data from memory
            stats_readmiss(cache_id);

            // Check if some other processor have issued request to same block of data
            locked_mutex = Port_Bus->check_ongoing_requests(cache_id, address, READ);
            Port_Bus->read(cache_id, address);
            wait(100);

            // Use LRU to replace
            int selected_way = lru_line(set_no);
            if(DEBUG_CACHE) Log << "\t@" << sc_time_stamp() << ": C" << cache_id << " selected_way : " <<selected_way << endl;

            cache_set[set_no].way[selected_way].data[word_in_line] = rand();
            intended_data = cache_set[set_no].way[selected_way].data[word_in_line];

            // Enable status bit, Set tag number
            cache_set[set_no].way[selected_way].status = 1;
            cache_set[set_no].way[selected_way].tag = tag_no;
            // Update LRU state
            update_lru_state(set_no, selected_way);
            found = true;

            // Release mutex held by current cache
            Port_Bus->release_mutex(cache_id, address);

            if(locked_mutex != -1){
                Log << "\t@" << sc_time_stamp() << ": Releasing mutex " << locked_mutex << " by " << cache_id << endl;
                // Mutex already locked for some other request. Release them.
                Port_Bus->release_mutex(locked_mutex, address);
            }
                   
            wait();
        }
        return intended_data;
    }
    else 
    {
        for (int i = 0; i < ASSOCIATIVITY; ++i)
        {
            if( (cache_set[set_no].way[i].status == 1) && (found == false) )
            {
                if (cache_set[set_no].way[i].tag == tag_no) 
                {
                    stats_writehit(cache_id);
                    Log << "\t@" << sc_time_stamp() << ": C" << cache_id << " Write hit \n";

                    locked_mutex = Port_Bus->check_ongoing_requests(cache_id, address, WRITE);
                    Port_Bus->write(cache_id, address, data);   // Place BusWr req
                    cache_set[set_no].way[i].data[word_in_line] = data;

                    wait(100); // Copy data to memory

                    // Enable status bit, update LRU
                    cache_set[set_no].way[i].status = 1;
                    update_lru_state(set_no, i);
                    found = true;
                    Port_Bus->release_mutex( cache_id, address);
                    if(locked_mutex != -1){
                        Log << "\t@" << sc_time_stamp() << ": Releasing mutex " << locked_mutex << " by " << cache_id << endl;
                        // Mutex already locked for some readX. Release them.
                        Port_Bus->release_mutex( locked_mutex, address);
                    }
                    wait();
                    break; 
                }   
            }
        }
        if (found == false)
        {
            // Write miss. Bring data from memory
            Log << "\t@" << sc_time_stamp() << ": C" << cache_id << " Write miss \n";
            stats_writemiss(cache_id);

            locked_mutex = Port_Bus->check_ongoing_requests(cache_id, address, READX);
            Port_Bus->readX(cache_id, address); // Place BusRdX req

            wait(100); // Bring data from memory

            // Use LRU to replace
            int selected_way = lru_line(set_no);
            if(DEBUG_CACHE) 
                Log << "\t@" << sc_time_stamp() << ": C" << cache_id << " selected_way to write: " << selected_way << endl;
            cache_set[set_no].way[selected_way].data[word_in_line] = rand();
            
            // Enable status bit
            cache_set[set_no].way[selected_way].status = 1;
            update_lru_state(set_no, selected_way);
            found = true;
            cache_set[set_no].way[selected_way].tag = tag_no;

            wait(100); // Copy latest data to memory
            Port_Bus->release_mutex(cache_id, address);

            if(locked_mutex != -1){
                Log << "\t@" << sc_time_stamp() << ": Releasing read mutex " << locked_mutex << " by " << cache_id << endl;
                // Mutex already locked for some read. Release them.
                Port_Bus->release_mutex(locked_mutex, address);
            }
            
            wait();
        }
        return 0;
    }
}

int Cache::lru_line(int set_no){
    int present_state = cache_set[set_no].lru;

    // Check if any way has invalid data
    for (int i = 0; i < ASSOCIATIVITY; ++i)
    {
        if(cache_set[set_no].way[i].status == 0){
            return i;
        }
    }

    if(DEBUG_CACHE) Log << "present_state : " << present_state << endl;

    if( (present_state & 11) == 0)
    {
        return 0;
    } else if( (present_state & 11) == 8 )
    {
        return 1;
    } else if ( (present_state & 19) == 2 )
    {
        return 2;
    } else if ( (present_state & 19) == 18 )
    {
        return 3;
    } else if ( (present_state & 37) == 1 )
    {
        return 4;
    } else if ( (present_state & 37) == 33 )
    {
        return 5;
    } else if ( (present_state & 69 ) == 5 )
    {
        return 6;
    } else if ( (present_state & 69 ) == 69 )
    {
        return 7;
    }
    else {
        if(DEBUG) cout << "\n---------- Error in finding out LRU line ---------- \n";
        Log << "\n---------- Error in finding out LRU line ---------- \n";
        return -1;  
    }
}

void Cache::update_lru_state(int set_no, int j){
    if(DEBUG_CACHE) Log << "update_lru_state : " << j << endl;
    switch(j) {
        case 0: // L0 is replaced. Next state ---1-11
            cache_set[set_no].lru = cache_set[set_no].lru | 11;
            break;
        case 1: // L1 is replaced. Next state ---0-11
            cache_set[set_no].lru = cache_set[set_no].lru | 3; //-----11
            cache_set[set_no].lru = cache_set[set_no].lru & 119; // ---0---
            break;
        case 2: // L2 is replaced. Next state --1--01
            cache_set[set_no].lru = cache_set[set_no].lru | 17; // --1---1
            cache_set[set_no].lru = cache_set[set_no].lru & 125; // -----0-
            break;
        case 3: // L3 is replaced. Next state --0--01
              cache_set[set_no].lru = cache_set[set_no].lru | 1; // ------1
              cache_set[set_no].lru = cache_set[set_no].lru & 109; // -----0-
              break;
        case 4: // L4 is replaced. Next state -1--1-0
              cache_set[set_no].lru = cache_set[set_no].lru | 36; // -1--1--
              cache_set[set_no].lru = cache_set[set_no].lru & 126; // ------0
              break;
        case 5: // L5 is replaced. Next state -0--1-0
              cache_set[set_no].lru = cache_set[set_no].lru | 4; // ----1--
              cache_set[set_no].lru = cache_set[set_no].lru & 94; // -0----0
              break;
        case 6: // L6 is replaced. Next state 1---0-0
              cache_set[set_no].lru = cache_set[set_no].lru | 64; // 1------
              cache_set[set_no].lru = cache_set[set_no].lru & 122; // ----0-0
              break;
        case 7: // L7 is replaced. Next state 0---0-0    
              cache_set[set_no].lru = cache_set[set_no].lru & 58; // 0---0-0   
              break;
        default:
              if(DEBUG) cout << "\n---------- Error in updating next LRU state ---------- \n";
              Log << "\n---------- Error in updating next LRU state ---------- \n";
              break;
    }
    if(DEBUG_CACHE) Log << "updated state : " << cache_set[set_no].lru << endl;
}

void Cache::invalidate_my_copy(int address){
    int set_no = (address & 4064) >> 5 ;      // 0b111111100000 = 4064
    unsigned int tag_no  = (address & 4294963200U) >> 12; // 0xfffff000 = 4294963200
    bool found = false;

    for (int i = 0; i < ASSOCIATIVITY; ++i)
    {
        if( (cache_set[set_no].way[i].status == 1) && (found == false) )
        {
            if( cache_set[set_no].way[i].tag == tag_no )
            {
                cache_set[set_no].way[i].status = 0;
                Log << "\t@" << sc_time_stamp() << ": Cache " <<cache_id << " invalidated a copy of " << address << "\n";
                found = true;
                desired_addresses++;
            }
        }
    }
}

/* Bus class, provides a way to share one memory in multiple CPU + Caches. */
class Bus : public Bus_if, public sc_module {
    public:
        /* Ports andkkk  vb Signals. */
        sc_in<bool> Port_CLK;
        sc_out<Cache::Function> Port_BusValid;
        sc_out<int> Port_BusWriter;

        sc_signal_rv<32> Port_BusAddr;

        /* Bus mutex. */
        sc_mutex bus;

        typedef struct 
        {
            sc_mutex get_access;
            int address;
            Req operation;
        }requests;

        requests *cache_requests;

        /* Variables. */
        long waits;
        long reads;
        long writes;
        long readXs;
        long consistency_waits;

    public:
        /* Constructor. */
        SC_CTOR(Bus) {
            /* Handle Port_CLK to simulate delay */
            sensitive << Port_CLK.pos();

            // Initialize some bus properties
            Port_BusAddr.write("ZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZ");

            /* Update variables. */
            waits = 0;
            reads = 0;
            writes = 0;
            readXs = 0;
            consistency_waits = 0; 

            cache_requests = new requests[num_cpus];
            for(unsigned int i=0; i<num_cpus; ++i){
                cache_requests[i].address = -1;
                cache_requests[i].operation = INVALID;
            }
        }

        ~Bus() {
            delete[] cache_requests;
        }

         virtual int check_ongoing_requests(int writer, int addr, Req opr) 
         {
            bool acquired_lock = false;
            int i;
            Req pending_req;
            
            for (i = 0; i < (int) num_cpus; ++i)
            {
                if (cache_requests[i].address == addr)
                {   
                    // Check if any request is issued for the same address by other processors
                    pending_req = cache_requests[i].operation;
                    if(DEBUG_BUS)
                        Log << "\t@" << sc_time_stamp() << ": Address_match, pending req: " << pending_req << "present : " << opr << endl; 

                    // Check for request sequence to decide if the current request have to wait
                    if( (pending_req == READ && (opr == READX || opr == WRITE)) ||
                        (pending_req == READX && (opr == READX || opr == READ)) ){
                        // Wait till other caches request is completed.
                        Log << "\t@" << sc_time_stamp() << ": Cache " << i << " has issued a req. Wait till it completes\n";
                        while(cache_requests[i].get_access.trylock() == -1){
                            consistency_waits++;
                            wait();
                        }
                        cache_requests[i].address = addr;
                        acquired_lock = true;
                        Log << "\t@" << sc_time_stamp() << ": Locked mutex " << writer << " addr " << addr;
                        break;
                    } 
                    
                }
            }
            if(acquired_lock == false){
                i = -1;
            }
            return i;
        }

        virtual void release_mutex(int writer, int addr) {
            // Request is processed completely. Release the mutex
            if( cache_requests[writer].address == addr ){
                cache_requests[writer].get_access.unlock();
                cache_requests[writer].address = -1;
                cache_requests[writer].operation = INVALID;
                if(DEBUG_BUS)
                    Log << "\t@" << sc_time_stamp() << ": release_mutex fn : mutex " << writer << " Addr: " << addr << endl;
            }
            else {
                Log <<"\t@" << sc_time_stamp() << ": Error releasing mutex " << writer << "address " << addr << endl;
            }
        }

        /* Perform a read access to memory addr for CPU #writer. */
        virtual bool read(int writer, int addr)
        {
            // Lock mutex and update address, operation field.
            while(cache_requests[writer].get_access.trylock() == -1){
                consistency_waits++;
                wait();
            }
            cache_requests[writer].address = addr;
            cache_requests[writer].operation = READ;

            if(DEBUG_BUS)
                Log << "\t@" << sc_time_stamp() << ": mutex " << writer << " locked with READ req\n";

            /* Try to get exclusive lock on bus. */
            while(bus.trylock() == -1){
                /* Wait when bus is in contention. */
                waits++;
                wait();
            }
            if(DEBUG_BUS)
                Log << "\t@" << sc_time_stamp() << ": C" << writer << " locked bus with READ req\n";
            /* Update number of bus accesses. */
            reads++;

            /* Set lines. */
            Port_BusAddr.write(addr);
            Port_BusWriter.write(writer);
            Port_BusValid.write(Cache::F_READ);

            /* Wait for everyone to recieve. */
            wait();

            /* Reset. */
            Port_BusValid.write(Cache::F_INVALID);
            Port_BusAddr.write("ZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZ");
            bus.unlock();

            if(DEBUG_BUS)
                Log << "\t@" << sc_time_stamp() << ": C" << writer << " released bus from READ req\n";

            return(true);
        };

        /* Write action to memory, need to know the writer, address and data. */
        virtual bool write(int writer, int addr, int data){
            // Lock mutex and update address, operation field.
            while(cache_requests[writer].get_access.trylock() == -1){
                consistency_waits++;
                wait();
            }
            cache_requests[writer].address = addr;
            cache_requests[writer].operation = WRITE;

            if(DEBUG_BUS) 
                Log << "\t@" << sc_time_stamp() << ": locked mutex " << writer << " addr " <<addr << endl;

            /* Try to get exclusive lock on the bus. */
            while(bus.trylock() == -1){
                waits++;
                wait();
            }

            if(DEBUG_BUS)
                Log << "@" << sc_time_stamp() << ": C" << writer << " locked bus with WRITE req\n";

            /* Update number of accesses. */
            writes++;

            /* Set. */
            Port_BusAddr.write(addr);
            Port_BusWriter.write(writer);
            Port_BusValid.write(Cache::F_WRITE);

            /* Wait for everyone to recieve. */
            wait();

            /* Reset. */
            Port_BusValid.write(Cache::F_INVALID);
            Port_BusAddr.write("ZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZ");
            bus.unlock();

            if(DEBUG_BUS)
                Log << "@" << sc_time_stamp() << ": C" << writer << " released bus from WRITE req\n";

            return(true);
        }

        /* Perform a read access to memory addr for CPU #writer. */
        virtual bool readX(int writer, int addr){

            // Lock mutex and update address, operation field. 
            while(cache_requests[writer].get_access.trylock() == -1){
                consistency_waits++;
                wait();
            }
            cache_requests[writer].address = addr;
            cache_requests[writer].operation = READX;

            if(DEBUG_BUS) 
                Log << "\t@" << sc_time_stamp() << ": locked readx mutex " << writer << " addr " <<addr << endl;
            /* Try to get exclusive lock on bus. */
            while(bus.trylock() == -1){
                /* Wait when bus is in contention. */
                waits++;
                wait();
            }

            if(DEBUG_BUS)
                Log << "\t@" << sc_time_stamp() << ": C" << writer << " locked bus with READX req\n";

            /* Update number of bus accesses. */
            readXs++;

            /* Set lines. */
            Port_BusAddr.write(addr);
            Port_BusWriter.write(writer);
            Port_BusValid.write(Cache::F_READX); // 3 -> read exclusive

            /* Wait for everyone to recieve. */
            wait();

            /* Reset. */
            Port_BusValid.write(Cache::F_INVALID); // 0 -> invalid
            Port_BusAddr.write("ZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZ");
            bus.unlock();

            if(DEBUG_BUS)
                Log << "\t@" << sc_time_stamp() << ": C" << writer << " released bus from READX req\n";

            return(true);
        };

        /* Bus output. */
        void output(){
            /* Write output as specified in the assignment. */
            double avg = (double)waits / double(reads + writes + readXs);
            long double exe_time = sc_time_stamp().value()/1000;
            printf("\n 2. Main memory access rates\n");
            printf("    Bus had %ld reads and %ld writes and %ld readX.\n", reads, writes, readXs);
            printf("    A total of %ld accesses.\n", reads + writes + readXs);
            printf("\n 3. Average time for bus acquisition\n");
            printf("    There were %ld waits for the bus.\n", waits);
            printf("    Average waiting time per access: %f cycles.\n", avg);
            printf("\n 4. There were %ld waits to maintain data consistency ", consistency_waits);
            printf("\n 5. %d addresses found while snooping bus requests ", desired_addresses);
            printf("\n 6. Total execution time is %ld ns, Avg per-mem-access time is %Lf ns\n", (long int)exe_time, exe_time/ double(reads + writes + readXs));
        }
};

SC_MODULE(CPU) 
{
    public:
        sc_in<bool>                Port_CLK;
        sc_in<Cache::RetCode>     Port_MemDone;
        sc_out<Cache::Function>   Port_MemFunc;
        sc_out<int>                Port_MemAddr;
        sc_inout_rv<32>            Port_MemData;
        int cpu_id;

        sc_out< sc_uint<2> > Mem_Func_Trace;

        SC_CTOR(CPU) 
        {
            SC_THREAD(execute);
            sensitive << Port_CLK.pos();
            dont_initialize();
        }

    private:
        void execute() 
        {

            TraceFile::Entry tr_data;
            Cache::Function f;  //Cache::Function f;
            int addr, data;

            while(!tracefile_ptr->eof())
            {
                if(!tracefile_ptr->next(cpu_id, tr_data)){
                    cerr << "Error reading trace for CPU" << endl;
                    Log << "Error reading trace for CPU" << endl;
                    break;
                }

                addr = tr_data.addr;

                switch(tr_data.type){
                    case TraceFile::ENTRY_TYPE_READ:
                        Log << "@" << sc_time_stamp() << ": P" << cpu_id << ": Read from " << addr << endl;
                        f = Cache::F_READ;
                        break;
                    case TraceFile::ENTRY_TYPE_WRITE:
                        Log << "@" << sc_time_stamp() << ": P" << cpu_id << ": Write to " << addr << endl;
                        f = Cache::F_WRITE;
                        break;
                    case TraceFile::ENTRY_TYPE_NOP:
                        f = Cache::F_INVALID;
                        break;
                    default:
                        cerr << "Error got invalid data from Trace" << endl;
                        Log << "@" << sc_time_stamp() << ": Error got invalid data from Trace" << endl;
                        exit(0);
                }

                if(f != Cache::F_INVALID){
                    Port_MemAddr.write(addr);
                    Port_MemFunc.write(f);

                    (f == Cache::F_READ) ? (Mem_Func_Trace.write(1)) : (Mem_Func_Trace.write(2));

                    if (f == Cache::F_WRITE) 
                    {
                        data = rand();
                        Port_MemData.write(data);
                        wait();
                        Port_MemData.write("ZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZ");
                    }

                    wait(Port_MemDone.value_changed_event());
                               
                }
                // Advance one cycle in simulated time 
                wait();
            }

            --pending_processors;
            if(pending_processors == 0){
                if(DEBUG) Log << "@" << sc_time_stamp() << ": Terminating simulation : " 
                    << sc_time_stamp().value()/1000 ;
                sc_stop();
            }
        
        }
};


int sc_main(int argc, char* argv[])
{
        // Get the tracefile argument and create Tracefile object
        // This function sets tracefile_ptr and num_cpus
        init_tracefile(&argc, &argv);

        // supress warnings & multiple driver issue
        sc_report_handler::set_actions(SC_ID_MORE_THAN_ONE_SIGNAL_DRIVER_, SC_DO_NOTHING);
        sc_report_handler::set_actions( SC_ID_LOGIC_X_TO_BOOL_, SC_LOG);
        sc_report_handler::set_actions( SC_ID_VECTOR_CONTAINS_LOGIC_VALUE_, SC_LOG);

        // Initialize statistics counters
        stats_init();

        pending_processors = num_cpus;
        desired_addresses = 0;
        
        // Log file 
        Log.open("Results.log");
        Log  << "\n\n------ Num CPU : " << num_cpus << " ------\n\n";

        /* Create clock. */
        sc_clock clk;

        /* Create Bus and TraceFile Syncronizer. */
        Bus  bus("bus");

         sc_trace_file *wf =sc_create_vcd_trace_file("WaveForms");

        /* Connect to Clock. */
        bus.Port_CLK(clk);

        /* Cpu and Cache pointers. */
        Cache* cache[num_cpus];
        CPU* cpu[num_cpus];

        /* Signals Cache/CPU. */
        sc_buffer<Cache::Function>  sigMemFunc[num_cpus];
        sc_buffer<Cache::RetCode>   sigMemDone[num_cpus];
        sc_signal_rv<32>            sigMemData[num_cpus];
        sc_signal<int>              sigMemAddr[num_cpus];

        /* Signals Chache/Bus. */
        sc_signal<int>              sigBusWriter;
        sc_signal<Cache::Function>  sigBusValid;

        sc_signal< sc_uint<2> > Sig_Mem_Func_Trace[num_cpus];

        /* General Bus signals. */
        bus.Port_BusWriter(sigBusWriter);
        bus.Port_BusValid(sigBusValid);

        /* Create and connect all caches and cpu's. */
        for(unsigned int i = 0; i < num_cpus; i++)
        {
            /* Each process should have a unique string name. */
            char name_cache[12];
            char name_cpu[12];

            char mem_func[15];
            char mem_data[15];
            char mem_done[15];
            char mem_addr[15];

            /* Use number in unique string name. */
            //name_cache << "cache_" << i;
            //name_cpu   << "cpu_"   << i;
            sprintf(name_cache, "cache_%d", i);
            sprintf(name_cpu, "cpu_%d", i);

            /* Create CPU and Cache. */
            cache[i] = new Cache(name_cache);
            cpu[i] = new CPU(name_cpu);

            /* Set ID's. */
            cpu[i]->cpu_id = i;
            cache[i]->cache_id = i;
            cache[i]->snooping = true;

            /* Cache to Bus. */
            cache[i]->Port_BusAddr(bus.Port_BusAddr);
            cache[i]->Port_BusWriter(sigBusWriter);
            cache[i]->Port_BusValid(sigBusValid);
            cache[i]->Port_Bus(bus);

            /* Cache to CPU. */
            cache[i]->Port_Func(sigMemFunc[i]);
            cache[i]->Port_Addr(sigMemAddr[i]);
            cache[i]->Port_Data(sigMemData[i]);
            cache[i]->Port_Done(sigMemDone[i]);

            /* CPU to Cache. */
            cpu[i]->Port_MemFunc(sigMemFunc[i]);
            cpu[i]->Port_MemAddr(sigMemAddr[i]);
            cpu[i]->Port_MemData(sigMemData[i]);
            cpu[i]->Port_MemDone(sigMemDone[i]);

            cpu[i]->Mem_Func_Trace(Sig_Mem_Func_Trace[i]);

            sprintf(mem_func, "sigMemFunc_%d", i);
            sprintf(mem_data, "sigMemData%d", i);
            sprintf(mem_done, "sigMemDone%d", i);
            sprintf(mem_addr, "sigMemAddr%d", i);

            sc_trace(wf, Sig_Mem_Func_Trace[i], mem_func); // 01 -> Read operation. 10 -> Write operation
            sc_trace(wf, sigMemAddr[i], mem_addr);
            sc_trace(wf, sigMemData[i], mem_data);
            //sc_trace(wf, sigMemDone[i], mem_done);

            /* Set Clock */
            cache[i]->Port_CLK(clk);
            cpu[i]->Port_CLK(clk);
        }

        // Dump the desired signals
        sc_trace(wf, clk, "clock");
        sc_trace(wf, bus.Port_BusAddr, "Bus_Addr");
        sc_trace(wf, bus.Port_BusWriter, "Port_BusWriter");
        //sc_trace(wf, sigBusValid, "sigBusValid");

        /* Start Simulation. */
        sc_start();
        stats_print();
        bus.output();

        // Close the Log file
        Log.close();
        sc_close_vcd_trace_file(wf);

        return 1;
}


