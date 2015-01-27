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
#define DEBUG 0

static const int MEM_SIZE = 32768;   // 32kB = 32 x 1024 Byte
static const int LINE_SIZE = 32;     // 32B Line size
static const int ASSOCIATIVITY = 8;  // 8 way set assosiative
static const int N_SET = MEM_SIZE/(LINE_SIZE*ASSOCIATIVITY); // 128 Lines

// Log file for the results
ofstream Log;

// Global command line arguments
int g_argc;
char** g_argv;

// Cache line should have  
// Tag, Status bit and Data
typedef struct {
    int data[8];
    unsigned int status :1;
    unsigned int tag : 20;     
}l1_cache_way;

// We need LRU for each set
typedef struct {
    l1_cache_way way[ASSOCIATIVITY];
    unsigned int lru : 7;
}l1_cache_set;

SC_MODULE(Memory) 
{

public:
    enum Function 
    {
        FUNC_READ,
        FUNC_WRITE
    };

    enum RetCode 
    {
        RET_READ_DONE,
        RET_WRITE_DONE,
    };

    sc_in<bool>     Port_CLK;
    sc_in<Function> Port_Func;
    sc_in<int>      Port_Addr;
    sc_out<RetCode> Port_Done;
    sc_inout_rv<32> Port_Data;

    int cache_lookup(int address, int mode);
    int lru_line(int set_no);
    void update_lru_state(int set_no, int j);

    SC_CTOR(Memory) 
    {
        SC_THREAD(execute);
        sensitive << Port_CLK.pos();
        dont_initialize();

        // Init cache
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

    ~Memory() 
    {
        delete cache_set;
    }

private:

    l1_cache_set *cache_set;
    int data;

    void execute() 
    {
        while (true)
        {
            wait(Port_Func.value_changed_event());

            Function f = Port_Func.read();
            int addr   = Port_Addr.read();
            data   = 0;
            int cache_data;

            if (f == FUNC_WRITE) 
            {
                data = Port_Data.read().to_int();
            }

            if (f == FUNC_READ) 
            {
                cache_data = cache_lookup(addr, READ_MODE);
                Port_Data.write(cache_data);
                Port_Done.write( RET_READ_DONE );
                wait();
                Port_Data.write("ZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZ");
            }
            else
            {
                cache_data = cache_lookup(addr, WRITE_MODE);
                if(DEBUG) cout << "MEM wrote " << data <<" at [" << addr << "]\n";
                Port_Done.write( RET_WRITE_DONE );
            }
        }
    }
}; 

// Member function definitions
int Memory::cache_lookup(int address, int mode){

    int set_no = (address & 4064) >> 5 ;      // 0b111111100000 = 4064
    unsigned int tag_no  = (address & 4294963200U) >> 12; // 0xfffff000 = 4294963200
    int word_in_line = (address & 28) >> 2;   // 0b11100 = 28
    int intended_data = 0;
    bool found = false;
    int pid = 0;
    // Read mode
        

    if(DEBUG) cout << "cache lookup. Line no: " << set_no << endl;
    if (mode == 0) 
    {
        if(DEBUG) cout << "Read mode \n";
        for (int i = 0; i < ASSOCIATIVITY; ++i)
        {
            if( (cache_set[set_no].way[i].tag == tag_no) && (found == false) ){
                // Tag line is present
                if (cache_set[set_no].way[i].status == 1)
                {
                    Log << "Tag is present. Read hit \n";
                    // Valid data
                    intended_data = cache_set[set_no].way[i].data[word_in_line];
                    stats_readhit(pid);
                    found = true;
                    wait();
                    break;
                }
                else {
                    // Read miss. Bring data from memory
                    stats_readmiss(pid);

                    Log << "Tag found. Read miss \n";
                    wait(100);

                    // Replace the invalid block
                    cache_set[set_no].way[i].data[word_in_line] = rand();
                    intended_data = cache_set[set_no].way[i].data[word_in_line];

                    // Enable status bit
                    cache_set[set_no].way[i].status = 1;
                    // Tag number already set
                    update_lru_state(set_no, i);
                    found = true;
                    break;
                }
            }
        }
        if (found == false)
        {
            if(DEBUG) cout << "Read miss: Tag: " << tag_no << endl;
            // Read miss. Bring data from memory
            stats_readmiss(pid);
            Log << "Read miss." << endl;

            wait(100);

            // Use LRU to replace
            int selected_way = lru_line(set_no);
            if(DEBUG) Log << "selected_way : " <<selected_way << endl;
            cache_set[set_no].way[selected_way].data[word_in_line] = rand();
            intended_data = cache_set[set_no].way[selected_way].data[word_in_line];
            // Enable status bit
            cache_set[set_no].way[selected_way].status = 1;
            // Set tag number
            cache_set[set_no].way[selected_way].tag = tag_no;
            // Update LRU state
            update_lru_state(set_no, selected_way);
            found = true;
        }
        return intended_data;
    }
    else {
        if(DEBUG) cout << "\n Write mode \n";

        for (int i = 0; i < ASSOCIATIVITY; ++i)
        {
            if( (cache_set[set_no].way[i].tag == tag_no) && (found == false) ){
                // Tag line is present
                // Invalid/Valid data: Replace the block.
                Log << "Write hit \n";
                cache_set[set_no].way[i].data[word_in_line] = data;

                //write hit
                stats_writehit(pid);

                // Enable status bit
                cache_set[set_no].way[i].status = 1;

                update_lru_state(set_no, i);
                found = true;
                wait();
                break;
            }
        }
        if (found == false)
        {
            // Write miss. Bring data from memory
            Log << "Write miss \n";
            stats_writemiss(pid);
            wait(100);

            // Use LRU to replace
            int selected_way = lru_line(set_no);
            if(DEBUG) Log << "selected_way to write: " <<selected_way << endl;
            cache_set[set_no].way[selected_way].data[word_in_line] = rand();
            
            // Enable status bit
            cache_set[set_no].way[selected_way].status = 1;

            // Update tag bit & LRU state table
            cache_set[set_no].way[selected_way].tag = tag_no;
            update_lru_state(set_no, selected_way);
            found = true;
            wait();
        }
        return 0;
    }
}

int Memory::lru_line(int set_no){
    int present_state = cache_set[set_no].lru;

    if(DEBUG) Log << "present_state : " << present_state << endl;

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
        return 0;  
    }
}

void Memory::update_lru_state(int set_no, int j){
    if(DEBUG) Log << "update_lru_state : " << j << endl;
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
    if(DEBUG) Log << "updated state : " << cache_set[set_no].lru << endl;
}

SC_MODULE(CPU) 
{
public:
    sc_in<bool>                Port_CLK;
    sc_in<Memory::RetCode>     Port_MemDone;
    sc_out<Memory::Function>   Port_MemFunc;
    sc_out<int>                Port_MemAddr;
    sc_inout_rv<32>            Port_MemData;

    SC_CTOR(CPU) 
    {
        SC_THREAD(execute);
        sensitive << Port_CLK.pos();
        dont_initialize();
    }

private:
    void execute() 
    {

        init_tracefile(&g_argc, &g_argv);
        stats_init();
        cout << "------ Num CPU : " << num_cpus << " ------\n";
        Log  << "------ Num CPU : " << num_cpus << " ------\n";

        TraceFile::Entry tr_data;
        Memory::Function f;
        int addr, data, pid =0 ;

        while(!tracefile_ptr->eof())
        {
            if(!tracefile_ptr->next(pid, tr_data)){
                cerr << "Error reading trace for CPU" << endl;
                Log << "Error reading trace for CPU" << endl;
                break;
            }

            addr = tr_data.addr;

            switch(tr_data.type){
                case TraceFile::ENTRY_TYPE_READ:
                    if(DEBUG) cout << "P" << pid << ": Read from " << addr << endl;
                    Log << "P" << pid << ": Read from " << addr << endl;
                    f = Memory::FUNC_READ;
                    break;
                case TraceFile::ENTRY_TYPE_WRITE:
                    if(DEBUG) cout << "P" << pid << ": Write to " << addr << endl;
                    Log << "P" << pid << ": Write to " << addr << endl;
                    f = Memory::FUNC_WRITE;
                    break;
                case TraceFile::ENTRY_TYPE_NOP:
                    break;
                default:
                    cerr << "Error got invalid data from Trace" << endl;
                    Log << "Error got invalid data from Trace" << endl;
                    exit(0);
            }

            if(DEBUG) Log << "\n*** Addr : " << addr << " ---- \n";
            //addr = addr % MEM_SIZE;
            Port_MemAddr.write(addr);
            Port_MemFunc.write(f);

            if (f == Memory::FUNC_WRITE) 
            {
                data = rand();
                Port_MemData.write(data);
                wait();
                Port_MemData.write("ZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZ");
            }
            wait(Port_MemDone.value_changed_event());
            // Advance one cycle in simulated time            
            wait();
        }

        if(DEBUG) Log << "Terminating simulation \n";
        sc_stop();
        
    }
};

int sc_main(int argc, char* argv[])
{
        // Log file 
        Log.open("Results.log");

    try
    {
        g_argv = argv;
        g_argc = argc;

        // Instantiate Modules
        Memory mem("main_memory");
        CPU    cpu("cpu");

        // Signals
        sc_buffer<Memory::Function> sigMemFunc;
        sc_buffer<Memory::RetCode>  sigMemDone;
        sc_signal<int>              sigMemAddr;
        sc_signal_rv<32>            sigMemData;

        // The clock that will drive the CPU and Memory
        sc_clock clk("clk", sc_time(1, SC_NS));

        // Connecting module ports with signals
        mem.Port_Func(sigMemFunc);
        mem.Port_Addr(sigMemAddr);
        mem.Port_Data(sigMemData);
        mem.Port_Done(sigMemDone);

        cpu.Port_MemFunc(sigMemFunc);
        cpu.Port_MemAddr(sigMemAddr);
        cpu.Port_MemData(sigMemData);
        cpu.Port_MemDone(sigMemDone);

        mem.Port_CLK(clk);
        cpu.Port_CLK(clk);

        sc_trace_file *wf =sc_create_vcd_trace_file("L1_WaveForms");
        // Dump the desired signals
        sc_trace(wf, clk, "clock");
        sc_trace(wf, sigMemDone, "Memory_Done");
        sc_trace(wf, sigMemAddr, "Memory_address");
        sc_trace(wf, sigMemAddr, "Memory_address");

        cout << "Running (press CTRL+C to interrupt)... " << endl;
 
        // Start Simulation
        sc_start();

        
        // Close the Log file
        Log.close();

        sc_close_vcd_trace_file(wf);

        stats_print();
    }

    catch (exception& e)
    {
        cerr << e.what() << endl;
    }
    return 0;
}
