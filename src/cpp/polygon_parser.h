#pragma once

#include <string>
#include <vector>
#include <memory>
#include <fstream>
#include <parquet/arrow/reader.h>
#include <arrow/api.h>

namespace liquidity {

class PolygonParser {
public:
    PolygonParser();
    ~PolygonParser();

    // Parse Polygon.io trades flat file (Parquet format)
    std::vector<TradeData> parse_trades_file(const std::string& filepath);
    
    // Parse Polygon.io quotes flat file (Parquet format)
    std::vector<QuoteData> parse_quotes_file(const std::string& filepath);
    
    // Stream large files
    class TradeStream {
    public:
        TradeStream(const std::string& filepath, size_t batch_size = 1000000);
        bool next_batch(std::vector<TradeData>& batch);
        
    private:
        std::unique_ptr<parquet::arrow::FileReader> reader_;
        int current_row_group_;
        int total_row_groups_;
        size_t batch_size_;
    };

    // Parse nanosecond timestamp from Polygon format
    static int64_t parse_sip_timestamp(int64_t sip_timestamp);
    
    // Convert exchange code
    static char parse_exchange_code(int exchange_id);

private:
    std::shared_ptr<arrow::MemoryPool> memory_pool_;
    
    // Helper functions for Parquet reading
    std::shared_ptr<arrow::Table> read_parquet_table(const std::string& filepath);
    void extract_trade_columns(const arrow::Table& table, std::vector<TradeData>& trades);
    void extract_quote_columns(const arrow::Table& table, std::vector<QuoteData>& quotes);
};

} // namespace liquidity
