#include "ap_int.h"
#include <hls_stream.h>

#define MAX_WIDTH 1920

typedef ap_uint<128> axie4_t; // 128 bit unsigned integer, to use the AXI4 protocol

extern "C" {    // use C-style linkage for the functions enclosed within its scope
    void filter_kernel(
        axie4_t *input_image,   // 16 bytes of image data
        axie4_t *output_image,
        float filter[3][3],     // customizable 3x3 filter
        float filter_divisor,   // customizable normalization factor 
        int width, 
        int height,
        int channels
    ) {
        // AXI Master Interface, kernel initiates memory transfer
        // `offset=slave` base address of the buffers will be provided by the PS through register
        // `bundle=gmem0` group interface signals into bundle, each bundle become separate physical AXI interface in hardware
        // `max_read_burst_length=256` transfer up to 256 data words in a single transaction, each word is 16 bytes so 4096 bytes per burst
        // `num_read_outstanding=8` allows up to 8 read requests to be "in flight" simultaneously, allows for pipelining
        #pragma HLS INTERFACE m_axi port=input_image offset=slave bundle=gmem0 max_read_burst_length=256 max_write_burst_length=256 num_read_outstanding=8 num_write_outstanding=8
        #pragma HLS INTERFACE m_axi port=output_image offset=slave bundle=gmem1 max_read_burst_length=256 max_write_burst_length=256 num_read_outstanding=8 num_write_outstanding=8
        // AXI-Lite Slave Interface
        // lower bandwidth interface for controls
        // `bundle=control` creates one shared AXI-Lite interface for all control parameters, in hardware this becomes a set of memory-mapped registers
        // ... accessible from the PS
        // `port=control` used to signal completion
        #pragma HLS INTERFACE s_axilite port=filter bundle=control
        #pragma HLS INTERFACE s_axilite port=filter_divisor bundle=control
        #pragma HLS INTERFACE s_axilite port=width bundle=control
        #pragma HLS INTERFACE s_axilite port=height bundle=control
        #pragma HLS INTERFACE s_axilite port=channels bundle=control
        #pragma HLS INTERFACE s_axilite port=return bundle=control

        // cache the filter in local memory for better performance
        float filter_local[3][3];
        // `complete` array completely partitioned
        // `dim=0` all dimensions of the array should be partitioned
        #pragma HLS ARRAY_PARTITION  variable=filter_local complete dim=0
        // before: 3x3 filter array would be implemented as a single BRAM, only one or two array element could be accessed
        // ... in a single clock cycle
        // after: each of the 9 elements in the filter matrix becomes a separate register, all 9 values can be accessed simultaneously
        // in a single clock cycle

        // copy filter to local memory
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                #pragma HLS UNROLL
                // before: reuses the same hardware for each iteration
                // after: all loop iterations execute concurrently
                filter_local[i][j] = filter[i][j];
            }
        }

        // store the filter divisor locally
        float divisor_local = filter_divisor;

        // 16 bytes per axie4 word
        const int elements_per_axie4 = 16;
        
        // line buffer to store 3 rows of the image
        unsigned char line_buffer[3][MAX_WIDTH][3];
        // partition the 3 rows of the image, for simultaneous access
        #pragma HLS ARRAY_PARTITION variable=line_buffer complete dim=1
        // partition the 3 rows of the image, for simultaneous access
        #pragma HLS ARRAY_PARTITION variable=line_buffer complete dim=3

        unsigned char window[3][3][3];
        #pragma HLS ARRAY_PARTITION variable=window complete dim=0

        // loop label
        init_line_buffer:
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < width; j++) {
                for (int k = 0; k < channels; k++) {
                    #pragma HLS PIPELINE
                    line_buffer[i][j][k] = 0;
                }
            }
        }

        process_rows:
        for (int row = 0; row < height; row++) {
            process_cols:
            for (int col = 0; col < width; col++) {
                // shift window left
                for (int i = 0; i < 3; i++) {
                    for (int j = 0; j < 2; j++) {
                        for (int k = 0; k < channels; k++) {
                            #pragma HLS UNROLL
                            window[i][j][k] = window[i][j+1][k];
                        }
                    }
                }
                
                // shift line buffer up, processing new row
                if (col == 0) {
                    shift_lines:
                    for (int i = 0; i < 2; i++) {
                        for (int j = 0; j < width; j++) {
                            for (int k = 0; k < channels; k++) {
                                #pragma HLS PIPELINE
                                line_buffer[i][j][k] = line_buffer[i+1][j][k];
                            }
                        }
                    }
                }

                // read new line into the buffer using the Axie-4 interface
                if (row < height) {
                    read_line:
                    for (int j = 0; j < width; j += elements_per_axie4) {
                        int axie4_idx = (row * width + j) * channels / elements_per_axie4;
                        axie4_t axie4_data = input_image[axie4_idx];
                        
                        // extract individual bytes from Axie-4 data
                        for (int x = 0; x < elements_per_axie4 && j + x < width; x++) {
                            for (int k = 0; k < channels && (j * channels + k) < elements_per_axie4; k++) {
                                #pragma HLS PIPELINE
                                // extract byte from 128-bit Axie-4 data
                                unsigned char pixel_val = (unsigned char) (axie4_data >> 8 * (x * channels + k));
                                if (row < height - 1 && j + x < width) {
                                    line_buffer[2][j + x][k] = pixel_val;
                                } else {
                                    line_buffer[2][j + x][k] = 0;
                                }
                            }
                        }
                    }
                }

                // update the right column of the window
                for (int i = 0; i < 3; i++) {
                    for (int k = 0; k < channels; k++) {
                        #pragma HLS UNROLL
                        if (col < width - 1) {
                            window[i][2][k] = line_buffer[i][col+1][k];
                        } else {
                            window[i][2][k] = 0;
                        }
                    }
                }

                // put the current pixel value in the center of the window
                for (int k = 0; k < channels; k++) {
                    #pragma HLS UNROLL
                    window[1][1][k] = line_buffer[1][col][k];
                }

                // --------------------------------------------------------------------
                // Modified boundary logic for 'reflect' mode:
                // We no longer skip or just copy for boundary pixels. Instead, we manually
                // reflect the row/col indices inside the 3x3 region. This means we do the
                // full convolution even at row=0, col=0, etc., but we "mirror" any -1 or
                // height/width overflow indices. 
                // 
                // We do it in a single block for all pixels. We'll ignore the old boundary
                // 'else if' block that just copied the original pixel. 
                // --------------------------------------------------------------------

                // buffer for output pixel
                unsigned char output_pixel[3];
                #pragma HLS ARRAY_PARTITION variable=output_pixel complete dim=0

                // We'll compute the filtered sum for each channel, reflecting boundary if needed
                for (int ch = 0; ch < channels; ch++) {
                    #pragma HLS UNROLL
                    float sum = 0.0f;

                    // We want a 3x3 around (row,col). For each offset (i2, j2) in [-1, 0, 1],
                    // we reflect if we go out of bounds.
                    for (int i2 = 0; i2 < 3; i2++) {
                        for (int j2 = 0; j2 < 3; j2++) {
                            #pragma HLS UNROLL
                            // compute the "actual" row/col for the window
                            // relative to the center pixel at (row, col)
                            int rr = row + (i2 - 1);
                            int cc = col + (j2 - 1);

                            // reflect row index if out of bounds
                            if (rr < 0) {
                                rr = -rr;                        // e.g., row=-1 => row=1
                            } else if (rr >= height) {
                                rr = (2 * height - rr - 2);      // e.g., row=height => row=height-2
                            }

                            // reflect col index if out of bounds
                            if (cc < 0) {
                                cc = -cc;                        // e.g., col=-1 => col=1
                            } else if (cc >= width) {
                                cc = (2 * width - cc - 2);
                            }

                            // Now we have a "reflected" rr, cc inside [0, height-1], [0, width-1].
                            // Next, we must figure out which row in line_buffer corresponds to 'rr'.
                            // Because line_buffer[0] is row-1, line_buffer[1] is row, line_buffer[2] is row+1
                            // for the *current* row. But we are streaming, so let's compute the offset:
                            int line_idx = rr - (row - 1);
                            // line_idx==0 => rr==row-1
                            // line_idx==1 => rr==row
                            // line_idx==2 => rr==row+1
                            // but if rr < row-1 or rr>row+1, we reflect again. 
                            // For a 3x3 kernel, we only need row±1 or col±1 for reflection, 
                            // so we do a clamp. 
                            if (line_idx < 0) {
                                line_idx = 0; // clamp or reflect further
                            }
                            if (line_idx > 2) {
                                line_idx = 2;
                            }

                            // same for col: we have cc but we can only access line_buffer[..][cc][ch] if cc in range
                            // so we clamp cc to [0, width-1]
                            if (cc < 0) {
                                cc = 0;
                            }
                            if (cc >= width) {
                                cc = width - 1;
                            }

                            // now read from line_buffer
                            float px_val = (float) line_buffer[line_idx][cc][ch];
                            // multiply by filter
                            sum += px_val * filter_local[i2][j2];
                        }
                    } // end of 3x3 loop

                    // apply divisor  and clamp result
                    int final_val = (int)(sum / divisor_local);
                    if (final_val < 0)   final_val = 0;
                    if (final_val > 255) final_val = 255;
                    output_pixel[ch] = (unsigned char)final_val;
                } // end channels loop

                // now write to output
                int pixel_idx = (row * width + col) * channels;
                int axie4_idx = pixel_idx / elements_per_axie4;
                int byte_offset = pixel_idx % elements_per_axie4;

                // get current output value, since we only need to modify a portion of it
                axie4_t axie4_data = output_image[axie4_idx];

                for (int k = 0; k < channels && (byte_offset + k) < elements_per_axie4; k++) {
                    #pragma HLS UNROLL
                    // clear
                    axie4_data &= ~(axie4_t(0xFF) << (8 * (byte_offset + k)));
                    // set
                    axie4_data |= (axie4_t(output_pixel[k])) << (8 * (byte_offset + k));
                }

                // write back
                output_image[axie4_idx] = axie4_data;

                // if pixel spans two Axie-4 words, handle the overflow
                if (byte_offset + channels > elements_per_axie4) {
                    int remaining_channels = byte_offset + channels - elements_per_axie4;
                    axie4_t next_axie4_data = output_image[axie4_idx + 1];
                    
                    for (int k = 0; k < remaining_channels; k++) {
                        #pragma HLS UNROLL
                        int channel_idx = elements_per_axie4 - byte_offset + k;
                        next_axie4_data &= ~(axie4_t(0xFF) << (8 * k));
                        next_axie4_data |= (axie4_t(output_pixel[channel_idx]) << (8 * k));
                    }

                    // write back
                    output_image[axie4_idx + 1] = next_axie4_data;
                }
            } // end col loop
        } // end row loop
    }
}
