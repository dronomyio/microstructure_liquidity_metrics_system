// ==================== src/cpp/polygon_parser.cpp ====================
#include "polygon_parser.h"
#include <arrow/io/file.h>
#include <parquet/arrow/reader.h>
#include <arrow/compute/api.h>
#include <iostream>

namespace liquidity {

PolygonParser::PolygonParser() {
    memory_pool_ = arrow::default_memory_pool();
}

PolygonParser::~PolygonParser() {}

std::vector<TradeData> PolygonParser::parse_trades_file(const std::string& filepath) {
    std::vector<TradeData> trades;
    
    try {
        auto table = read_parquet_table(filepath);
        if (table) {
            extract_trade_columns(*table, trades);
        }
    } catch (const std::exception& e) {
        std::cerr << "Error parsing trades file: " << e.what() << std::endl;
    }
    
    return trades;
}

std::vector<QuoteData> PolygonParser::parse_quotes_file(const std::string& filepath) {
    std::vector<QuoteData> quotes;
    
    try {
        auto table = read_parquet_table(filepath);
        if (table) {
            extract_quote_columns(*table, quotes);
        }
    } catch (const std::exception& e) {
        std::cerr << "Error parsing quotes file: " << e.what() << std::endl;
    }
    
    return quotes;
}

std::shared_ptr<arrow::Table> PolygonParser::read_parquet_table(const std::string& filepath) {
    std::shared_ptr<arrow::io::ReadableFile> infile;
    PARQUET_ASSIGN_OR_THROW(
        infile,
        arrow::io::ReadableFile::Open(filepath, memory_pool_)
    );
    
    std::unique_ptr<parquet::arrow::FileReader> reader;
    PARQUET_THROW_NOT_OK(
        parquet::arrow::OpenFile(infile, memory_pool_, &reader)
    );
    
    std::shared_ptr<arrow::Table> table;
    PARQUET_THROW_NOT_OK(reader->ReadTable(&table));
    
    return table;
}

void PolygonParser::extract_trade_columns(
    const arrow::Table& table,
    std::vector<TradeData>& trades
) {
    // Get columns
    auto timestamp_col = table.GetColumnByName("sip_timestamp");
    auto price_col = table.GetColumnByName("price");
    auto size_col = table.GetColumnByName("size");
    auto exchange_col = table.GetColumnByName("exchange");
    auto conditions_col = table.GetColumnByName("conditions");
    
    if (!timestamp_col || !price_col || !size_col) {
        throw std::runtime_error("Required columns not found in trades file");
    }
    
    int64_t num_rows = table.num_rows();
    trades.reserve(num_rows);
    
    // Extract data from Arrow arrays
    for (int64_t i = 0; i < num_rows; ++i) {
        TradeData trade;
        
        // Simplified extraction (real implementation would handle chunks properly)
        auto timestamp_array = std::static_pointer_cast<arrow::Int64Array>(
            timestamp_col->chunk(0)
        );
        auto price_array = std::static_pointer_cast<arrow::DoubleArray>(
            price_col->chunk(0)
        );
        auto size_array = std::static_pointer_cast<arrow::DoubleArray>(
            size_col->chunk(0)
        );
        
        trade.timestamp_ns = timestamp_array->Value(i);
        trade.price = price_array->Value(i);
        trade.volume = size_array->Value(i);
        trade.conditions = 0; // Simplified
        trade.exchange = 'N'; // Simplified
        
        trades.push_back(trade);
    }
}

void PolygonParser::extract_quote_columns(
    const arrow::Table& table,
    std::vector<QuoteData>& quotes
) {
    // Get columns
    auto timestamp_col = table.GetColumnByName("sip_timestamp");
    auto bid_price_col = table.GetColumnByName("bid_price");
    auto ask_price_col = table.GetColumnByName("ask_price");
    auto bid_size_col = table.GetColumnByName("bid_size");
    auto ask_size_col = table.GetColumnByName("ask_size");
    
    if (!timestamp_col || !bid_price_col || !ask_price_col) {
        throw std::runtime_error("Required columns not found in quotes file");
    }
    
    int64_t num_rows = table.num_rows();
    quotes.reserve(num_rows);
    
    // Extract data (simplified)
    for (int64_t i = 0; i < num_rows; ++i) {
        QuoteData quote;
        
        auto timestamp_array = std::static_pointer_cast<arrow::Int64Array>(
            timestamp_col->chunk(0)
        );
        auto bid_price_array = std::static_pointer_cast<arrow::DoubleArray>(
            bid_price_col->chunk(0)
        );
        auto ask_price_array = std::static_pointer_cast<arrow::DoubleArray>(
            ask_price_col->chunk(0)
        );
        auto bid_size_array = std::static_pointer_cast<arrow::Int32Array>(
            bid_size_col->chunk(0)
        );
        auto ask_size_array = std::static_pointer_cast<arrow::Int32Array>(
            ask_size_col->chunk(0)
        );
        
        quote.timestamp_ns = timestamp_array->Value(i);
        quote.bid_price = bid_price_array->Value(i);
        quote.ask_price = ask_price_array->Value(i);
        quote.bid_size = bid_size_array->Value(i);
        quote.ask_size = ask_size_array->Value(i);
        quote.bid_exchange = 'N';
        quote.ask_exchange = 'N';
        
        quotes.push_back(quote);
    }
}

int64_t PolygonParser::parse_sip_timestamp(int64_t sip_timestamp) {
    // SIP timestamp is already in nanoseconds
    return sip_timestamp;
}

char PolygonParser::parse_exchange_code(int exchange_id) {
    // Simplified mapping
    switch (exchange_id) {
        case 1: return 'A';  // NYSE American
        case 2: return 'B';  // NASDAQ BX
        case 3: return 'N';  // NYSE
        case 4: return 'P';  // NYSE Arca
        case 5: return 'Q';  // NASDAQ
        case 8: return 'X';  // NASDAQ PSX
        default: return 'U'; // Unknown
    }
}

// TradeStream implementation
PolygonParser::TradeStream::TradeStream(const std::string& filepath, size_t batch_size)
    : current_row_group_(0), batch_size_(batch_size) {
    
    std::shared_ptr<arrow::io::ReadableFile> infile;
    PARQUET_ASSIGN_OR_THROW(
        infile,
        arrow::io::ReadableFile::Open(filepath)
    );
    
    PARQUET_THROW_NOT_OK(
        parquet::arrow::OpenFile(infile, arrow::default_memory_pool(), &reader_)
    );
    
    total_row_groups_ = reader_->num_row_groups();
}

bool PolygonParser::TradeStream::next_batch(std::vector<TradeData>& batch) {
    if (current_row_group_ >= total_row_groups_) {
        return false;
    }
    
    batch.clear();
    
    // Read row group
    std::shared_ptr<arrow::Table> table;
    PARQUET_THROW_NOT_OK(
        reader_->ReadRowGroup(current_row_group_, &table)
    );
    
    // Extract trades (simplified)
    // ... extraction logic similar to extract_trade_columns ...
    
    current_row_group_++;
    return !batch.empty();
}

} // namespace liquidity
